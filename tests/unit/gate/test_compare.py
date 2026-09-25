"""evaluate_gate: (baseline runs, current runs, policy) -> per-config verdicts.

A pure function -- no file I/O, no retriever, no corpus -- so the anti-cheat
asymmetry from ADR-0010 (golden set and corpus are a hard fail; config is a
warning) is tested directly against hand-built ScoringReports rather than
through a slow, real evaluation run.
"""

from __future__ import annotations

from groundtruth.gate.compare import evaluate_config, evaluate_gate
from groundtruth.gate.policy import GatePolicy
from groundtruth.scoring.models import LatencySummary, MetricSummary, ScoringReport

LATENCY = LatencySummary(embed_ms=1, dense_ms=1, lexical_ms=0, fuse_ms=0, rerank_ms=0, total_ms=2)


def report(
    config_name: str = "dense_512",
    *,
    config_hash: str = "cfg-1",
    golden_set_hash: str = "golden-1",
    corpus_manifest_hash: str = "corpus-1",
    recall: float = 0.85,
    ndcg: float = 0.6,
) -> ScoringReport:
    return ScoringReport(
        config_name=config_name,
        config_hash=config_hash,
        golden_set_hash=golden_set_hash,
        corpus_manifest_hash=corpus_manifest_hash,
        code_version="0.1.0",
        run_fingerprint=f"fp-{config_name}",
        n_queries=90,
        n_skipped=0,
        overall={
            "recall": {
                10: MetricSummary(mean=recall, ci_low=recall - 0.05, ci_high=recall + 0.05, n=90)
            },
            "ndcg": {10: MetricSummary(mean=ndcg, ci_low=ndcg - 0.05, ci_high=ndcg + 0.05, n=90)},
        },
        by_category={},
        by_origin={},
        latency=LATENCY,
    )


def policy(**overrides: object) -> GatePolicy:
    defaults: dict[str, object] = {
        "primary_metric": "recall_at_10",
        "gated_configs": ("dense_512",),
        "max_absolute_drop": {"recall_at_10": 0.010, "ndcg_at_10": 0.015},
        "require_golden_set_hash_match": True,
        "require_corpus_hash_match": True,
        "config_hash_mismatch": "warn",
    }
    defaults.update(overrides)
    return GatePolicy(**defaults)  # type: ignore[arg-type]


class TestNoBaseline:
    def test_a_config_with_no_baseline_is_skipped_not_failed(self):
        result = evaluate_config("dense_512", {}, {"dense_512": report()}, policy())
        assert result.status == "skipped_no_baseline"
        assert result.passed


class TestMissingFromCurrentRun:
    def test_a_config_measured_before_but_absent_now_is_a_hard_failure(self):
        base = {"dense_512": report()}
        result = evaluate_config("dense_512", base, {}, policy())
        assert result.status == "failed"
        assert not result.passed
        assert any("missing" in f for f in result.identity_failures)


class TestIdentityAntiCheat:
    def test_an_unchanged_golden_set_and_corpus_pass(self):
        base = {"dense_512": report()}
        current = {"dense_512": report()}
        result = evaluate_config("dense_512", base, current, policy())
        assert result.passed
        assert result.identity_failures == ()

    def test_a_changed_golden_set_hash_is_a_hard_failure(self):
        base = {"dense_512": report(golden_set_hash="golden-1")}
        current = {"dense_512": report(golden_set_hash="golden-2")}
        result = evaluate_config("dense_512", base, current, policy())
        assert result.status == "failed"
        assert any("golden set" in f for f in result.identity_failures)

    def test_a_changed_corpus_hash_is_a_hard_failure(self):
        base = {"dense_512": report(corpus_manifest_hash="corpus-1")}
        current = {"dense_512": report(corpus_manifest_hash="corpus-2")}
        result = evaluate_config("dense_512", base, current, policy())
        assert result.status == "failed"
        assert any("corpus" in f for f in result.identity_failures)

    def test_a_changed_config_hash_only_warns_by_default(self):
        # Changing the config IS the workflow -- failing here would make the
        # gate useless within a day.
        base = {"dense_512": report(config_hash="cfg-1")}
        current = {"dense_512": report(config_hash="cfg-2")}
        result = evaluate_config("dense_512", base, current, policy())
        assert result.passed
        assert result.identity_failures == ()

    def test_a_changed_config_hash_fails_when_the_policy_says_fail(self):
        base = {"dense_512": report(config_hash="cfg-1")}
        current = {"dense_512": report(config_hash="cfg-2")}
        result = evaluate_config("dense_512", base, current, policy(config_hash_mismatch="fail"))
        assert result.status == "failed"

    def test_golden_set_and_corpus_are_not_gated_when_the_policy_says_so(self):
        base = {"dense_512": report(golden_set_hash="golden-1", corpus_manifest_hash="corpus-1")}
        current = {"dense_512": report(golden_set_hash="golden-2", corpus_manifest_hash="corpus-2")}
        result = evaluate_config(
            "dense_512",
            base,
            current,
            policy(require_golden_set_hash_match=False, require_corpus_hash_match=False),
        )
        assert result.passed


class TestMetricRegression:
    def test_a_drop_within_tolerance_passes(self):
        base = {"dense_512": report(recall=0.850)}
        current = {"dense_512": report(recall=0.845)}  # -0.005, limit -0.010
        result = evaluate_config("dense_512", base, current, policy())
        assert result.passed
        assert result.regressions == ()

    def test_a_drop_exactly_at_the_limit_passes(self):
        base = {"dense_512": report(recall=0.850)}
        current = {"dense_512": report(recall=0.840)}  # exactly -0.010
        result = evaluate_config("dense_512", base, current, policy())
        assert result.passed

    def test_a_drop_past_the_limit_fails_and_names_the_metric(self):
        base = {"dense_512": report(recall=0.842)}
        current = {"dense_512": report(recall=0.617)}
        result = evaluate_config("dense_512", base, current, policy())
        assert result.status == "failed"
        (regression,) = [r for r in result.regressions if r.metric == "recall"]
        assert regression.k == 10
        assert regression.baseline_mean == 0.842
        assert regression.current_mean == 0.617
        assert "0.842" in str(regression)
        assert "0.617" in str(regression)

    def test_an_improvement_is_never_a_failure(self):
        base = {"dense_512": report(recall=0.80)}
        current = {"dense_512": report(recall=0.95)}
        result = evaluate_config("dense_512", base, current, policy())
        assert result.passed

    def test_ndcg_can_fail_while_recall_stays_within_tolerance(self):
        # This is the point of gating multiple metrics: a chunk-size change
        # can hurt ranking quality without hurting whether the right chunk
        # was found at all.
        base = {"dense_512": report(recall=0.85, ndcg=0.60)}
        current = {"dense_512": report(recall=0.845, ndcg=0.50)}
        result = evaluate_config("dense_512", base, current, policy())
        assert result.status == "failed"
        assert [r.metric for r in result.regressions] == ["ndcg"]

    def test_a_metric_the_policy_gates_but_neither_run_reports_is_a_failure(self):
        base_report = report()
        current_report = report()
        result = evaluate_config(
            "dense_512",
            {"dense_512": base_report},
            {"dense_512": current_report},
            policy(max_absolute_drop={"mrr_at_10": 0.02}),
        )
        assert result.status == "failed"
        assert "mrr_at_10" in result.unreported_metrics


class TestEvaluateGate:
    def test_returns_one_result_per_gated_config_in_order(self):
        base = {"a": report("a"), "b": report("b")}
        current = {"a": report("a"), "b": report("b")}
        results = evaluate_gate(base, current, policy(gated_configs=("a", "b")))
        assert [r.config_name for r in results] == ["a", "b"]

    def test_a_config_not_yet_built_does_not_block_the_others(self):
        base = {"dense_512": report()}
        current = {"dense_512": report()}
        results = evaluate_gate(
            base, current, policy(gated_configs=("dense_512", "hybrid_512_rerank"))
        )
        by_name = {r.config_name: r for r in results}
        assert by_name["dense_512"].passed
        assert by_name["hybrid_512_rerank"].status == "skipped_no_baseline"
