"""Reciprocal Rank Fusion.

::

    score(d) = sum over arms a where d appears of  1 / (k + rank_a(d))

with ``rank`` 1-based.

RRF reads **ranks and discards scores**, which is the entire reason it is used
here. The dense arm produces cosine similarities in [-1, 1]; BM25 produces
unbounded non-negative scores whose scale depends on corpus statistics. Adding
them requires a normalization step, and every normalization step is a knob:
min-max normalization makes the fused score depend on the worst candidate
retrieved, and z-scoring makes it depend on the spread. Both would have to be
tuned, and tuning them would make the hybrid-vs-dense comparison a measurement
of tuning effort rather than of method.

``k`` defaults to 60 in the configs -- the value from Cormack et al. (2009),
used as published. It damps the difference between top ranks: at k=60, rank 1
is worth 1/61 and rank 2 is worth 1/62, so no single arm can dominate on the
strength of one confident hit.

An arm that does not return a document contributes nothing for it, rather than
a penalty. A document found by both arms therefore outranks one found
emphatically by a single arm, which is the behaviour hybrid retrieval is for.
"""

from __future__ import annotations

from collections.abc import Sequence

from groundtruth.index.models import ScoredChunk
from groundtruth.retrieval.ordering import order_candidates


class FusionError(Exception):
    """Candidate lists could not be fused."""


def reciprocal_rank_fusion(
    ranked_arms: Sequence[Sequence[ScoredChunk]], *, k: int
) -> tuple[ScoredChunk, ...]:
    """Fuse already-ordered candidate lists into one ranking.

    Each arm must already be in its own rank order -- ``rank`` is taken from
    position, so an unordered arm silently fuses the wrong numbers.
    """
    if k <= 0:
        raise FusionError(f"k must be positive, got {k}; k=0 makes rank 1 contribute 1/0")
    if not ranked_arms:
        raise FusionError("fusion needs at least one candidate list")

    fused: dict[str, float] = {}
    for arm_index, arm in enumerate(ranked_arms):
        seen: set[str] = set()
        for rank, candidate in enumerate(arm, start=1):
            if candidate.chunk_id in seen:
                raise FusionError(
                    f"duplicate chunk {candidate.chunk_id!r} in arm {arm_index}; "
                    f"one document would be paid twice for a single piece of evidence"
                )
            seen.add(candidate.chunk_id)
            fused[candidate.chunk_id] = fused.get(candidate.chunk_id, 0.0) + 1.0 / (k + rank)

    return order_candidates(ScoredChunk(chunk_id, score) for chunk_id, score in fused.items())
