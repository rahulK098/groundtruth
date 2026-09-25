"""Recall@k, MRR@k, nDCG@k against worked arithmetic (ADR-0004).

Every non-trivial case here is computed by hand in the docstring or the
comment beside the assertion, so a bug shows up as a wrong number against a
number someone can re-check with a pencil -- which is the whole point of
hand-rolling these instead of importing a scorer library.
"""

from __future__ import annotations

import math

from groundtruth.golden.models import RelevanceLabel
from groundtruth.retrieval.models import PassageScores, RetrievedPassage
from groundtruth.scoring.metrics import mrr_at_k, ndcg_at_k, recall_at_k

DOC = "cl-1"


def passage(chunk_id: str, start: int, end: int, rank: int, doc_id: str = DOC) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text="x" * (end - start),
        char_start=start,
        char_end=end,
        rank=rank,
        scores=PassageScores(),
    )


def label(start: int, end: int, gain: int, doc_id: str = DOC) -> RelevanceLabel:
    return RelevanceLabel(
        doc_id=doc_id, char_start=start, char_end=end, gain=gain, quote="x" * (end - start)
    )


# A 200-char span at [0:200) needs 100 chars of overlap to be covered.
RELEVANT_SPAN = label(0, 200, gain=3)
# A second, non-overlapping 200-char span at [1000:1200).
SECOND_SPAN = label(1000, 1200, gain=2)


class TestSkippedQueries:
    """A query with no gain>=2 span is excluded from the mean, never scored 0.0.

    GoldenPair's own validator already forbids this in practice; these tests
    exercise the metric functions directly against a raw label list, which is
    the only way to reach the branch at all.
    """

    def test_recall_returns_none_with_no_relevant_span(self):
        only_partial = [label(0, 200, gain=1)]
        assert recall_at_k([passage("c1", 0, 200, 1)], only_partial, k=10) is None

    def test_mrr_returns_none_with_no_relevant_span(self):
        only_partial = [label(0, 200, gain=1)]
        assert mrr_at_k([passage("c1", 0, 200, 1)], only_partial, k=10) is None

    def test_ndcg_returns_none_with_no_relevant_span(self):
        only_partial = [label(0, 200, gain=1)]
        assert ndcg_at_k([passage("c1", 0, 200, 1)], only_partial, k=10) is None

    def test_an_empty_retrieval_result_is_not_a_skip(self):
        # Zero passages is a real (bad) result, not an unscoreable query.
        assert recall_at_k([], [RELEVANT_SPAN], k=10) == 0.0


class TestRecallAtK:
    def test_the_relevant_span_covered_at_rank_1(self):
        result = [passage("c1", 0, 200, 1)]
        assert recall_at_k(result, [RELEVANT_SPAN], k=10) == 1.0

    def test_the_relevant_span_not_covered_by_anything_retrieved(self):
        result = [passage("c1", 5000, 5200, 1)]
        assert recall_at_k(result, [RELEVANT_SPAN], k=10) == 0.0

    def test_covered_only_past_the_k_cutoff_does_not_count(self):
        result = [passage("miss", 5000, 5200, 1), passage("c1", 0, 200, 2)]
        assert recall_at_k(result, [RELEVANT_SPAN], k=1) == 0.0
        assert recall_at_k(result, [RELEVANT_SPAN], k=2) == 1.0

    def test_two_relevant_spans_half_covered_is_one_half(self):
        result = [passage("c1", 0, 200, 1)]  # covers RELEVANT_SPAN only
        assert recall_at_k(result, [RELEVANT_SPAN, SECOND_SPAN], k=10) == 0.5

    def test_a_partial_gain_span_gain_1_never_counts_toward_recall(self):
        partial = label(0, 200, gain=1)
        result = [passage("c1", 0, 200, 1)]
        # Only RELEVANT_SPAN (gain 3) is in the denominator; the partial span
        # is covered too, but is not gain>=2, so it neither helps nor hurts.
        assert recall_at_k(result, [RELEVANT_SPAN, partial], k=10) == 1.0

    def test_span_level_not_chunk_level(self):
        # Two overlapping chunks both covering the same span: this must not
        # be double-counted, and the span is either covered or it is not.
        result = [passage("c1", 0, 150, 1), passage("c2", 50, 200, 2)]
        assert recall_at_k(result, [RELEVANT_SPAN], k=10) == 1.0


class TestMrrAtK:
    def test_first_covering_passage_at_rank_1_gives_rr_1(self):
        result = [passage("c1", 0, 200, 1)]
        assert mrr_at_k(result, [RELEVANT_SPAN], k=10) == 1.0

    def test_first_covering_passage_at_rank_3_gives_rr_one_third(self):
        result = [
            passage("miss1", 5000, 5200, 1),
            passage("miss2", 6000, 6200, 2),
            passage("c1", 0, 200, 3),
        ]
        assert mrr_at_k(result, [RELEVANT_SPAN], k=10) == 1.0 / 3.0

    def test_no_covering_passage_gives_rr_zero(self):
        result = [passage("miss", 5000, 5200, 1)]
        assert mrr_at_k(result, [RELEVANT_SPAN], k=10) == 0.0

    def test_covering_passage_past_k_gives_rr_zero_at_that_k(self):
        result = [passage("miss", 5000, 5200, 1), passage("c1", 0, 200, 2)]
        assert mrr_at_k(result, [RELEVANT_SPAN], k=1) == 0.0
        assert mrr_at_k(result, [RELEVANT_SPAN], k=2) == 0.5

    def test_mrr_sees_only_the_first_relevant_span_by_construction(self):
        # Both spans are covered, at ranks 1 and 2. MRR only reflects rank 1.
        result = [passage("c1", 0, 200, 1), passage("c2", 1000, 1200, 2)]
        assert mrr_at_k(result, [RELEVANT_SPAN, SECOND_SPAN], k=10) == 1.0


class TestNdcgAtK:
    def test_perfect_ranking_scores_1_0(self):
        # One span, gain 3, covered at rank 1: DCG == IDCG.
        result = [passage("c1", 0, 200, 1)]
        assert ndcg_at_k(result, [RELEVANT_SPAN], k=10) == 1.0

    def test_no_relevant_chunk_retrieved_scores_0_0(self):
        result = [passage("miss", 5000, 5200, 1)]
        assert ndcg_at_k(result, [RELEVANT_SPAN], k=10) == 0.0

    def test_worked_example_two_spans_reordered(self):
        """Two spans, gains 3 and 2 -- reversed from ideal order.

        IDCG@2 (ideal: gain 3 at rank 1, gain 2 at rank 2):
            (2**3 - 1)/log2(2) + (2**2 - 1)/log2(3)
          = 7/1.0 + 3/1.5849625007211562
          = 7.0 + 1.8927892607...
          = 8.8927892607...

        Actual ranking: gain-2 span covered at rank 1, gain-3 span at rank 2.
        DCG@2 = (2**2 - 1)/log2(2) + (2**3 - 1)/log2(3)
              = 3/1.0 + 7/1.5849625007211562
              = 3.0 + 4.4165184751...
              = 7.4165184751...

        nDCG@2 = DCG / IDCG = 7.4165184751... / 8.8927892607... = 0.834026...
        """
        gain3 = label(0, 200, gain=3)
        gain2 = label(1000, 1200, gain=2)
        result = [passage("c-gain2", 1000, 1200, 1), passage("c-gain3", 0, 200, 2)]

        idcg = (2**3 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
        dcg = (2**2 - 1) / math.log2(2) + (2**3 - 1) / math.log2(3)
        expected = dcg / idcg

        score = ndcg_at_k(result, [gain3, gain2], k=10)
        assert score is not None
        assert score == expected
        assert 0.83 < score < 0.84

    def test_span_deduplication_a_later_chunk_covering_the_same_span_earns_nothing(self):
        # Two chunks both cover the ONE span. The second must not be credited
        # again, or IDCG-independent redundancy would inflate the score past
        # 1.0 and break comparability across chunk sizes (ADR-0004).
        result = [passage("c1", 0, 150, 1), passage("c2", 50, 200, 2)]
        assert ndcg_at_k(result, [RELEVANT_SPAN], k=10) == 1.0

    def test_a_chunk_covering_several_uncredited_spans_earns_the_max_gain_and_credits_all(self):
        # One chunk spans both labeled spans at once (a large chunk over two
        # adjacent holdings). It should earn the higher gain and credit both,
        # leaving nothing for a later chunk to add.
        wide_span_a = label(0, 200, gain=2)
        wide_span_b = label(150, 350, gain=3)
        one_chunk = [passage("wide", 0, 350, 1)]

        idcg = (2**3 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
        # DCG: rank 1 covers both uncredited spans, credited at max gain (3).
        dcg = (2**3 - 1) / math.log2(2)
        expected = dcg / idcg

        score = ndcg_at_k(one_chunk, [wide_span_a, wide_span_b], k=10)
        assert score == expected

    def test_gains_beyond_k_are_excluded_from_idcg_too(self):
        # Three spans, only ndcg@1 requested: IDCG@1 uses only the single
        # highest gain, not all three.
        spans = [label(0, 100, gain=3), label(1000, 1100, gain=2), label(2000, 2100, gain=2)]
        result = [passage("c1", 0, 100, 1)]
        assert ndcg_at_k(result, spans, k=1) == 1.0


class TestMonotonicity:
    """Sanity properties that hold for any input, not just the worked cases."""

    def test_recall_is_non_decreasing_in_k(self):
        result = [
            passage("miss", 5000, 5200, 1),
            passage("c1", 0, 200, 2),
            passage("c2", 1000, 1200, 3),
        ]
        r1 = recall_at_k(result, [RELEVANT_SPAN, SECOND_SPAN], k=1)
        r2 = recall_at_k(result, [RELEVANT_SPAN, SECOND_SPAN], k=2)
        r3 = recall_at_k(result, [RELEVANT_SPAN, SECOND_SPAN], k=3)
        assert r1 is not None and r2 is not None and r3 is not None
        assert r1 <= r2 <= r3

    def test_ndcg_stays_within_zero_and_one(self):
        result = [passage("c1", 0, 200, 1), passage("c2", 1000, 1200, 2)]
        score = ndcg_at_k(result, [RELEVANT_SPAN, SECOND_SPAN], k=10)
        assert score is not None
        assert 0.0 <= score <= 1.0 + 1e-9
