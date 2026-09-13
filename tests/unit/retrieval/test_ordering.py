"""Deterministic ordering.

Ties are resolved once, here, and nowhere else. Every arm and the fusion step
funnel through this so a tie can never be broken two different ways in one
pipeline.
"""

from __future__ import annotations

import math
import random

import pytest

from groundtruth.index.models import ScoredChunk
from groundtruth.retrieval.ordering import OrderingError, order_candidates, take_top_n


class TestOrderCandidates:
    def test_sorts_by_score_descending(self):
        ordered = order_candidates(
            [ScoredChunk("a", 0.1), ScoredChunk("b", 0.9), ScoredChunk("c", 0.5)]
        )
        assert [c.chunk_id for c in ordered] == ["b", "c", "a"]

    def test_breaks_ties_by_chunk_id_ascending(self):
        ordered = order_candidates(
            [ScoredChunk("zeta", 1.0), ScoredChunk("alpha", 1.0), ScoredChunk("mid", 1.0)]
        )
        assert [c.chunk_id for c in ordered] == ["alpha", "mid", "zeta"]

    def test_input_order_cannot_change_the_output(self):
        # The property that licenses calling the harness reproducible: an
        # index that returned tied candidates in a different order on another
        # machine must still produce the same ranking.
        candidates = [ScoredChunk(f"chunk-{i:02d}", float(i % 3)) for i in range(30)]
        expected = order_candidates(candidates)

        shuffled = list(candidates)
        rng = random.Random(20240101)
        for _ in range(20):
            rng.shuffle(shuffled)
            assert order_candidates(shuffled) == expected

    def test_rejects_a_duplicate_chunk_id(self):
        # Two rows for one chunk would be counted twice by every metric.
        with pytest.raises(OrderingError, match="duplicate"):
            order_candidates([ScoredChunk("a", 1.0), ScoredChunk("a", 0.5)])

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_a_non_finite_score(self, bad: float):
        # NaN compares false against everything, so it would sort into an
        # arbitrary position and quietly destroy the ordering guarantee.
        with pytest.raises(OrderingError, match="finite"):
            order_candidates([ScoredChunk("a", 1.0), ScoredChunk("b", bad)])

    def test_empty_input_is_allowed(self):
        assert order_candidates([]) == ()


class TestTakeTopN:
    def test_truncates_after_ordering(self):
        top = take_top_n([ScoredChunk("a", 0.1), ScoredChunk("b", 0.9), ScoredChunk("c", 0.5)], 2)
        assert [c.chunk_id for c in top] == ["b", "c"]

    def test_returns_everything_when_n_exceeds_the_input(self):
        assert len(take_top_n([ScoredChunk("a", 1.0)], 10)) == 1

    def test_rejects_a_non_positive_n(self):
        with pytest.raises(OrderingError, match="positive"):
            take_top_n([ScoredChunk("a", 1.0)], 0)
