"""Deterministic ordering. Resolved once, here, and nowhere else.

Every arm and the fusion step funnel through this, so a tie can never be
broken two different ways inside one pipeline, and the scorer downstream
receives a total order containing no tie logic of its own.

**Ties break by ``chunk_id`` ascending.** The literature offers
expected-value-under-random-ties variants of nDCG that are arguably more
principled. They are rejected here for one reason: this project's product is
a number a reader can reproduce. A metric that moves between runs of identical
code cannot support the claim the harness exists to make. The tradeoff is
stated in the methodology rather than buried.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from groundtruth.index.models import ScoredChunk


class OrderingError(Exception):
    """A candidate list could not be put into a defined order."""


def order_candidates(candidates: Iterable[ScoredChunk]) -> tuple[ScoredChunk, ...]:
    """Sort by score descending, then ``chunk_id`` ascending.

    Rejects the two inputs that would silently destroy the guarantee:

    * a **non-finite score**, because NaN compares false against everything
      and would land in an arbitrary position;
    * a **duplicate chunk id**, because one chunk occupying two ranks is
      counted twice by every metric downstream.
    """
    materialized = tuple(candidates)

    seen: set[str] = set()
    for candidate in materialized:
        if not math.isfinite(candidate.score):
            raise OrderingError(
                f"chunk {candidate.chunk_id!r} has a non-finite score "
                f"({candidate.score!r}); scores must be finite or the ranking "
                f"is undefined"
            )
        if candidate.chunk_id in seen:
            raise OrderingError(
                f"duplicate chunk id {candidate.chunk_id!r} in a candidate list; "
                f"one chunk must occupy exactly one rank"
            )
        seen.add(candidate.chunk_id)

    return tuple(sorted(materialized, key=lambda c: (-c.score, c.chunk_id)))


def take_top_n(candidates: Iterable[ScoredChunk], n: int) -> tuple[ScoredChunk, ...]:
    """Order, then truncate.

    In that order, always. Truncating first would make the result depend on
    whatever order the candidates happened to arrive in.
    """
    if n <= 0:
        raise OrderingError(f"n must be positive, got {n}")
    return order_candidates(candidates)[:n]
