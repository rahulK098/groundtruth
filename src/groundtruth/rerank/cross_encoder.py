"""The real bge cross-encoder.

**Imported lazily**, for the same reason as the embedder: ``torch`` and
``sentence_transformers`` live in the ``models`` extra, which the gate never
installs. Only ``gt cache rerank`` constructs this class.
"""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import TYPE_CHECKING, Any, Final

from groundtruth.embedding.sentence_transformer import ModelExtraNotInstalledError
from groundtruth.rerank.protocol import RerankScores

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import CrossEncoder

#: Pinned for the same reason as the embedder's: thread count and batch size
#: move low-order bits, and here they also move the recorded latency -- so
#: both are written into the cache metadata.
DEFAULT_BATCH_SIZE: Final[int] = 16
DEFAULT_THREADS: Final[int] = 1

#: The model's own limit. A (query, 512-token chunk) pair exceeds it and is
#: truncated -- stated here rather than discovered, because it means the
#: reranker judges slightly less of each passage than the retriever indexed.
DEFAULT_MAX_LENGTH: Final[int] = 512


class CrossEncoderReranker:
    """Scores (query, passage) pairs with a pinned bge-reranker checkpoint on CPU."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        threads: int = DEFAULT_THREADS,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._batch_size = batch_size
        self._threads = threads
        self._max_length = max_length
        self._model = self._load()

    def _load(self) -> CrossEncoder:
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ModelExtraNotInstalledError(
                "The 'models' extra is not installed, so no rerank scores can be "
                "computed. This is expected in the gate environment, which serves "
                "committed scores only.\n\n"
                "To regenerate the rerank cache:\n"
                "    uv sync --frozen --extra dev --extra models"
            ) from exc

        torch.set_num_threads(self._threads)
        model: CrossEncoder = CrossEncoder(
            self._model_id,
            revision=self._revision,
            device="cpu",
            max_length=self._max_length,
        )
        return model

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def build_settings(self) -> dict[str, Any]:
        """Recorded in the rerank cache metadata; the latency figures depend on it."""
        return {
            "device": "cpu",
            "batch_size": self._batch_size,
            "torch_threads": self._threads,
            "max_length": self._max_length,
        }

    def score(self, query: str, passages: Sequence[str]) -> RerankScores:
        if not passages:
            return RerankScores(scores=(), compute_ms=0.0)

        started = perf_counter()
        raw = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        compute_ms = (perf_counter() - started) * 1000.0
        return RerankScores(scores=tuple(float(s) for s in raw), compute_ms=compute_ms)
