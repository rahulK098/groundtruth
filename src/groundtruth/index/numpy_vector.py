"""Exact cosine search over an in-memory matrix.

This is the dense arm the regression gate measures. It is exact -- one matrix
multiply, every candidate scored, no approximation anywhere -- which is the
property that makes a committed baseline meaningful. An ANN index would make
the number depend on index build order and search-time parameters, so a
"regression" could be an artifact of the index rather than of the change under
test. At roughly 15,000 chunks the exact scan takes single-digit milliseconds,
so there is nothing to buy by approximating.

FAISS was rejected for the same reason plus one more: it is a substantial
dependency for an operation NumPy expresses in one line.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from groundtruth.index.models import ScoredChunk


class VectorIndexError(Exception):
    """A vector index could not be built or searched."""


class NumpyVectorIndex:
    """Exact cosine similarity over a dense matrix of chunk vectors."""

    backend = "numpy"

    def __init__(self, chunk_ids: Sequence[str], vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            raise VectorIndexError(f"vectors must be a 2-D matrix, got shape {vectors.shape}")
        if len(chunk_ids) != vectors.shape[0]:
            raise VectorIndexError(
                f"{len(chunk_ids)} chunk ids but {vectors.shape[0]} vector rows; "
                f"they must align or every lookup returns another chunk's embedding"
            )
        if len(set(chunk_ids)) != len(chunk_ids):
            raise VectorIndexError("duplicate chunk ids: one chunk must own exactly one row")

        matrix = np.asarray(vectors, dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise VectorIndexError("vectors contain non-finite values")

        norms = np.linalg.norm(matrix, axis=1)
        if not norms.all():
            zero = int(np.argmin(norms))
            raise VectorIndexError(
                f"chunk {chunk_ids[zero]!r} has a zero vector; cosine similarity "
                f"against it is undefined and would rank as NaN"
            )

        # Sorting by chunk id at construction is what makes the tie-break free:
        # a *stable* sort on descending score then yields score desc, chunk_id
        # asc with no secondary key and no per-query string comparison.
        order = sorted(range(len(chunk_ids)), key=lambda i: chunk_ids[i])
        self._chunk_ids: tuple[str, ...] = tuple(str(chunk_ids[i]) for i in order)
        # Vectors are normalized once here rather than assumed to be unit
        # length. The committed cache is float16, so its stored norms are
        # 0.9998-1.0001; dividing by the true norm is what makes the reported
        # score an exact cosine rather than a very good approximation.
        self._matrix: np.ndarray = matrix[order] / norms[order][:, None]

    @property
    def count(self) -> int:
        return len(self._chunk_ids)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return self._chunk_ids

    @property
    def dimension(self) -> int:
        return int(self._matrix.shape[1])

    def search(self, query_vector: np.ndarray, top_n: int) -> tuple[ScoredChunk, ...]:
        """Return the ``top_n`` most similar chunks, deterministically ordered."""
        if top_n <= 0:
            raise VectorIndexError(f"top_n must be positive, got {top_n}")

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dimension:
            raise VectorIndexError(
                f"query has dimension {query.shape[0]} but the index holds "
                f"{self.dimension}-dimensional vectors"
            )
        if not np.isfinite(query).all():
            raise VectorIndexError("query vector contains non-finite values")

        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            raise VectorIndexError("query vector is zero; cosine similarity is undefined")

        scores = self._matrix @ (query / norm)

        # A full stable argsort, not argpartition. Partition selects *a* set of
        # top_n, so when candidates tie across the cut, which one lands inside
        # is an implementation detail rather than a stated rule -- and the tie
        # rule is exactly what makes this harness reproducible. Sorting ~15,000
        # floats costs about a millisecond, which buys nothing worth the loss.
        order = np.argsort(-scores, kind="stable")[:top_n]
        return tuple(ScoredChunk(self._chunk_ids[i], float(scores[i])) for i in order)
