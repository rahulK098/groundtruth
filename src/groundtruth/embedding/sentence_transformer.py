"""The real bge embedder.

**Imported lazily, on purpose.** ``torch`` and ``sentence_transformers`` live
in the ``models`` optional extra, which the gate never installs. Importing them
at module scope would make this file unimportable in the gate environment and
would drag ~2.5 GB into any environment that did install them eagerly.

A test asserts neither module is in ``sys.modules`` after the cache path is
imported, so the discipline is enforced rather than merely intended.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

#: Single-threaded, fixed batch. Both affect low-order bits of the output, so
#: they are pinned and recorded in the cache metadata -- a cache regenerated
#: with different settings may differ in the last decimal, and this is what
#: makes that diagnosable instead of mysterious.
DEFAULT_BATCH_SIZE: Final[int] = 32
DEFAULT_THREADS: Final[int] = 1


class ModelExtraNotInstalledError(Exception):
    """The optional `models` extra is not installed."""


class UnknownEmbeddingDimensionError(Exception):
    """The model does not report a fixed sentence-embedding width."""


class SentenceTransformerEmbedder:
    """Embeds with a pinned bge checkpoint on CPU."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        normalize: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        threads: int = DEFAULT_THREADS,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._normalize = normalize
        self._batch_size = batch_size
        self._threads = threads
        self._model = self._load()

    def _load(self) -> SentenceTransformer:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ModelExtraNotInstalledError(
                "The 'models' extra is not installed, so no embeddings can be "
                "computed. This is expected in the gate environment, which "
                "serves committed vectors only.\n\n"
                "To regenerate the cache:\n"
                "    uv sync --frozen --extra dev --extra models"
            ) from exc

        # Determinism knobs. Thread count changes reduction order, which moves
        # the last bits of every vector.
        torch.set_num_threads(self._threads)

        return SentenceTransformer(
            self._model_id,
            revision=self._revision,
            device="cpu",
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def dimension(self) -> int:
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            # Happens for architectures sentence-transformers cannot infer a
            # single output width for. Such a model cannot back a fixed-width
            # cache, so failing here beats writing a ragged store.
            raise UnknownEmbeddingDimensionError(
                f"{self._model_id} does not report a sentence embedding "
                f"dimension, so it cannot back a fixed-width embedding cache."
            )
        return int(dimension)

    @property
    def build_settings(self) -> dict[str, Any]:
        """Recorded in the cache metadata for reproducibility forensics."""
        return {
            "device": "cpu",
            "batch_size": self._batch_size,
            "torch_threads": self._threads,
            "compute_dtype": "float32",
        }

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
