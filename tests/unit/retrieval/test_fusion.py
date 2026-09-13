"""Reciprocal Rank Fusion.

RRF reads ranks and ignores scores, which is the whole reason it is used here:
a cosine similarity in [0, 1] and a BM25 score in [0, 30) cannot be added
without a calibration step that would itself need tuning and defending.
"""

from __future__ import annotations

import pytest

from groundtruth.index.models import ScoredChunk
from groundtruth.retrieval.fusion import FusionError, reciprocal_rank_fusion

# Ranks are 1-based, so with k = 60 the contributions are 1/61, 1/62, 1/63 ...
DENSE = (ScoredChunk("a", 0.90), ScoredChunk("b", 0.80), ScoredChunk("c", 0.70))
LEXICAL = (ScoredChunk("c", 14.0), ScoredChunk("a", 9.0), ScoredChunk("d", 2.0))


class TestKnownAnswer:
    """Hand-computed, k = 60.

    a: 1/61 + 1/62 = 0.0163934 + 0.0161290 = 0.0325225
    c: 1/63 + 1/61 = 0.0158730 + 0.0163934 = 0.0322665
    b: 1/62                                = 0.0161290
    d: 1/63                                = 0.0158730
    """

    def test_fused_order(self):
        fused = reciprocal_rank_fusion([DENSE, LEXICAL], k=60)
        assert [c.chunk_id for c in fused] == ["a", "c", "b", "d"]

    def test_fused_scores(self):
        fused = {c.chunk_id: c.score for c in reciprocal_rank_fusion([DENSE, LEXICAL], k=60)}
        assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
        assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)
        assert fused["b"] == pytest.approx(1 / 62)
        assert fused["d"] == pytest.approx(1 / 63)

    def test_a_document_ranked_first_by_one_arm_can_still_lose(self):
        # c leads the lexical list outright and still finishes second,
        # because a is supported by both arms. If this ever inverts, fusion
        # has stopped fusing.
        fused = reciprocal_rank_fusion([DENSE, LEXICAL], k=60)
        assert fused[0].chunk_id == "a"
        assert LEXICAL[0].chunk_id == "c"


class TestBehaviour:
    def test_incoming_scores_are_discarded(self):
        # Same ranks, wildly different scores: identical output.
        rescaled = tuple(ScoredChunk(c.chunk_id, c.score * 1000) for c in LEXICAL)
        assert reciprocal_rank_fusion([DENSE, LEXICAL], k=60) == reciprocal_rank_fusion(
            [DENSE, rescaled], k=60
        )

    def test_a_single_list_is_reordered_but_not_reweighted(self):
        fused = reciprocal_rank_fusion([DENSE], k=60)
        assert [c.chunk_id for c in fused] == ["a", "b", "c"]

    def test_arm_order_does_not_matter(self):
        assert reciprocal_rank_fusion([DENSE, LEXICAL], k=60) == reciprocal_rank_fusion(
            [LEXICAL, DENSE], k=60
        )

    def test_an_empty_arm_contributes_nothing(self):
        assert reciprocal_rank_fusion([DENSE, ()], k=60) == reciprocal_rank_fusion([DENSE], k=60)

    def test_ties_break_by_chunk_id_ascending(self):
        # Two documents at rank 1 of different arms score identically.
        left = (ScoredChunk("zulu", 1.0),)
        right = (ScoredChunk("alpha", 1.0),)
        fused = reciprocal_rank_fusion([left, right], k=60)
        assert [c.chunk_id for c in fused] == ["alpha", "zulu"]

    def test_a_smaller_k_sharpens_the_top_of_the_ranking(self):
        # k controls how much a top rank is worth relative to a deep one.
        # This is the knob's documented behaviour, so it is pinned.
        sharp = {c.chunk_id: c.score for c in reciprocal_rank_fusion([DENSE], k=1)}
        flat = {c.chunk_id: c.score for c in reciprocal_rank_fusion([DENSE], k=1000)}
        assert sharp["a"] / sharp["c"] > flat["a"] / flat["c"]


class TestValidation:
    def test_rejects_no_arms(self):
        with pytest.raises(FusionError, match="at least one"):
            reciprocal_rank_fusion([], k=60)

    def test_rejects_a_non_positive_k(self):
        # k = 0 makes the first rank contribute 1/0.
        with pytest.raises(FusionError, match="positive"):
            reciprocal_rank_fusion([DENSE], k=0)

    def test_rejects_a_duplicate_within_one_arm(self):
        # One document occupying two ranks of the same list would be paid
        # twice for a single piece of evidence.
        with pytest.raises(FusionError, match="duplicate"):
            reciprocal_rank_fusion([(ScoredChunk("a", 1.0), ScoredChunk("a", 0.5))], k=60)
