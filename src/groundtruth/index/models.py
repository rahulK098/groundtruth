"""The value object every index returns.

A slotted dataclass rather than a pydantic model, which is a deliberate
exception to the project's convention. Tens of thousands of these are created
per query sweep and none of them crosses a system boundary: they are produced
by an index and consumed by fusion, both of which are this project's own code.
Pydantic validation buys nothing there and costs roughly an order of magnitude
per instance.

The values that *do* cross a boundary -- what a caller or a result file sees --
are the pydantic models in :mod:`groundtruth.retrieval.models`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """One candidate and the score the arm that produced it assigned.

    The score is only ever comparable within the arm that produced it: a
    cosine similarity and a BM25 score share no scale. That is precisely why
    fusion consumes ranks rather than these numbers.
    """

    chunk_id: str
    score: float
