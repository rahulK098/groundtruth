"""Assembling a retriever for the evaluation path.

One function, and its job is to make the chunker, the embedding cache and the
two indexes agree. Every one of them is keyed by something the others must
match -- chunk ids, embedding cache keys, vector rows -- and assembling them
in more than one place is how those keys drift apart.

This builds the **evaluation path only**: NumPy and in-process BM25, no
Postgres, no network. The service path gets its own assembly in Phase 10, over
the same :class:`~groundtruth.retrieval.pipeline.Retriever`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from groundtruth.chunking.models import Chunk
from groundtruth.chunking.pipeline import chunk_corpus
from groundtruth.config.models import RetrievalConfig
from groundtruth.corpus.models import Document
from groundtruth.embedding.protocol import Embedder
from groundtruth.index.bm25 import Bm25Index
from groundtruth.index.numpy_vector import NumpyVectorIndex
from groundtruth.index.protocol import LexicalIndex
from groundtruth.retrieval.pipeline import Retriever


class UnsupportedBackendError(Exception):
    """A configuration asks for a backend this path cannot provide."""


def build_lexical_index(config: RetrievalConfig, chunks: tuple[Chunk, ...]) -> LexicalIndex | None:
    """Build the lexical arm, or nothing for a dense configuration."""
    lexical = config.lexical
    if lexical is None:
        return None

    if lexical.backend != "bm25":
        # Substituting BM25 for pg_fts would make a result file labelled
        # `pg_fts` a lie about what actually ran -- and the two differ in the
        # one respect the comparison exists to measure (ADR-0007).
        raise UnsupportedBackendError(
            f"lexical backend {lexical.backend!r} is not available on the "
            f"evaluation path; pg_fts is served by Postgres and is measured "
            f"through the service path"
        )

    return Bm25Index(
        tuple(chunk.chunk_id for chunk in chunks),
        tuple(chunk.text for chunk in chunks),
        k1=lexical.k1,
        b=lexical.b,
    )


def build_retriever(
    config: RetrievalConfig, documents: Iterable[Document], embedder: Embedder
) -> Retriever:
    """Chunk, embed and index a corpus, and return a retriever over it.

    ``embedder`` is normally a
    :class:`~groundtruth.embedding.cache.CachedOnlyEmbedder`, which serves
    committed vectors and raises rather than computing. That is what makes a
    model download during an evaluation run structurally impossible instead of
    merely discouraged (ADR-0003).
    """
    chunks = chunk_corpus(documents, config.chunking)
    if not chunks:
        raise UnsupportedBackendError(
            "the corpus produced no chunks; there is nothing to retrieve from"
        )

    # One call, so a cache-only embedder reports every missing string at once
    # rather than failing on the first and hiding the scale of the problem.
    vectors = np.asarray(embedder.embed([chunk.text for chunk in chunks]), dtype=np.float32)

    return Retriever(
        config,
        embedder=embedder,
        vector_index=NumpyVectorIndex(tuple(chunk.chunk_id for chunk in chunks), vectors),
        lexical_index=build_lexical_index(config, chunks),
        chunks={chunk.chunk_id: chunk for chunk in chunks},
    )
