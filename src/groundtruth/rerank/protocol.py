"""What the retriever needs from a reranker, and nothing more."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RerankScores:
    """Scores for one query's candidate batch, and what they cost to compute.

    ``compute_ms`` is carried with the scores rather than timed by the caller
    because for a cached reranker the two differ by three orders of
    magnitude: the lookup costs ~0 ms, the model it stands in for did not.
    The retriever reports this figure as the stage's latency (ADR-0012).
    """

    #: One score per passage, in the order the passages were given. Higher
    #: is more relevant; the scale is the model's own and is only ever
    #: compared within one batch.
    scores: tuple[float, ...]
    compute_ms: float


@runtime_checkable
class Reranker(Protocol):
    """Scores passages against a query. Identity is checked against the config."""

    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> RerankScores: ...
