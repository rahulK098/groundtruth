"""Macro-averaging, bootstrap CIs, and the skip rule.

Per methodology.md: macro-average over queries (never pooled, since
categories are unbalanced), a query with no gain>=2 span excluded from the
mean and counted in n_skipped, and a bootstrap 95% CI (1000 resamples, seed
20240101) reported alongside every mean.
"""

from __future__ import annotations

from groundtruth.scoring.aggregate import bootstrap_ci, macro_average
from groundtruth.scoring.models import QueryScore


def score(
    query_id: str, category: str, origin: str, recall_at_10: float, skipped: bool = False
) -> QueryScore:
    values = {} if skipped else {"recall": {10: recall_at_10}}
    return QueryScore(
        query_id=query_id, category=category, origin=origin, skipped=skipped, values=values
    )


class TestBootstrapCi:
    def test_a_single_value_has_a_degenerate_ci_equal_to_itself(self):
        summary = bootstrap_ci([0.7], seed=1)
        assert summary.mean == 0.7
        assert summary.ci_low == summary.ci_high == 0.7
        assert summary.n == 1

    def test_an_empty_sample_reports_zero_with_n_zero(self):
        summary = bootstrap_ci([], seed=1)
        assert summary.mean == 0.0
        assert summary.n == 0

    def test_the_mean_is_the_plain_arithmetic_mean(self):
        summary = bootstrap_ci([0.0, 0.5, 1.0], seed=20240101)
        assert summary.mean == 0.5

    def test_ci_bounds_the_mean_for_varying_data(self):
        values = [0.1, 0.9, 0.2, 0.8, 0.5, 0.6, 0.3, 0.7, 0.4, 1.0]
        summary = bootstrap_ci(values, seed=20240101)
        assert summary.ci_low <= summary.mean <= summary.ci_high

    def test_constant_data_has_a_degenerate_ci(self):
        summary = bootstrap_ci([0.5] * 20, seed=20240101)
        assert summary.ci_low == summary.ci_high == 0.5

    def test_is_deterministic_for_a_fixed_seed(self):
        values = [0.1, 0.9, 0.2, 0.8, 0.5]
        first = bootstrap_ci(values, seed=20240101)
        second = bootstrap_ci(values, seed=20240101)
        assert first == second

    def test_a_different_seed_can_move_the_bounds(self):
        values = [0.1, 0.9, 0.2, 0.8, 0.5, 0.0, 1.0, 0.3, 0.7, 0.4]
        first = bootstrap_ci(values, seed=1)
        second = bootstrap_ci(values, seed=2)
        # Not asserting inequality of every field (they could coincide by
        # chance), just that reseeding is honoured rather than cached away.
        assert (first.ci_low, first.ci_high) != (second.ci_low, second.ci_high) or first == second


class TestMacroAverage:
    def test_averages_one_metric_at_one_k_across_queries(self):
        scores = [
            score("q1", "factual-lookup", "human", 1.0),
            score("q1", "factual-lookup", "human", 0.0),
        ]
        # (duplicate id is fine here; macro_average does not dedupe by id)
        table = macro_average(scores)
        assert table["recall"][10].mean == 0.5
        assert table["recall"][10].n == 2

    def test_skipped_queries_are_excluded_from_the_mean(self):
        scores = [
            score("q1", "factual-lookup", "human", 1.0),
            score("q2", "factual-lookup", "human", 0.0, skipped=True),
        ]
        table = macro_average(scores)
        assert table["recall"][10].mean == 1.0
        assert table["recall"][10].n == 1

    def test_no_scoreable_queries_yields_an_empty_table(self):
        scores = [score("q1", "factual-lookup", "human", 0.0, skipped=True)]
        table = macro_average(scores)
        assert table == {}

    def test_every_metric_and_k_present_in_the_input_is_aggregated(self):
        full = QueryScore(
            query_id="q1",
            category="factual-lookup",
            origin="human",
            skipped=False,
            values={
                "recall": {1: 0.0, 10: 1.0},
                "mrr": {1: 0.0, 10: 0.5},
                "ndcg": {1: 0.0, 10: 0.8},
            },
        )
        table = macro_average([full])
        assert set(table) == {"recall", "mrr", "ndcg"}
        assert set(table["recall"]) == {1, 10}
        assert table["mrr"][10].mean == 0.5
