"""Serializing a ScoringReport to and from JSON.

Round-trip is the property that matters: whatever `gt run` writes must be
exactly what `gt baseline bless` and the gate read back, with every hash and
every number intact.
"""

from __future__ import annotations

import json
from pathlib import Path

from groundtruth.scoring.models import LatencySummary, MetricSummary, QueryScore, ScoringReport
from groundtruth.scoring.report_io import (
    RUNS_DIRNAME,
    read_baseline,
    read_report,
    report_from_dict,
    report_to_dict,
    write_baseline,
    write_run,
)


def sample_report(config_name: str = "dense_512") -> ScoringReport:
    return ScoringReport(
        config_name=config_name,
        config_hash="abc123",
        golden_set_hash="sha256:golden",
        corpus_manifest_hash="sha256:corpus",
        code_version="0.1.0",
        run_fingerprint="fingerprint123",
        n_queries=90,
        n_skipped=0,
        overall={
            "recall": {10: MetricSummary(mean=0.82, ci_low=0.75, ci_high=0.88, n=90)},
            "mrr": {10: MetricSummary(mean=0.6, ci_low=0.5, ci_high=0.7, n=90)},
        },
        by_category={
            "factual-lookup": {
                "recall": {10: MetricSummary(mean=0.9, ci_low=0.8, ci_high=1.0, n=20)}
            }
        },
        by_origin={
            "human": {"recall": {10: MetricSummary(mean=0.85, ci_low=0.7, ci_high=0.95, n=25)}}
        },
        latency=LatencySummary(
            embed_ms=1.0, dense_ms=0.6, lexical_ms=0.0, fuse_ms=0.0, rerank_ms=0.0, total_ms=1.6
        ),
    )


class TestDictRoundTrip:
    def test_report_survives_a_round_trip_unchanged(self):
        original = sample_report()
        restored = report_from_dict(report_to_dict(original))
        assert restored == original

    def test_the_dict_form_is_json_serializable(self):
        # Every dict key must be a string -- int k-values need coercion.
        payload = report_to_dict(sample_report())
        json.dumps(payload)  # raises on anything non-serializable

    def test_a_report_with_no_breakdowns_round_trips(self):
        empty = sample_report().__class__(
            config_name="x",
            config_hash="h",
            golden_set_hash="g",
            corpus_manifest_hash="c",
            code_version="v",
            run_fingerprint="f",
            n_queries=0,
            n_skipped=0,
            overall={},
            by_category={},
            by_origin={},
            latency=LatencySummary(0, 0, 0, 0, 0, 0),
        )
        assert report_from_dict(report_to_dict(empty)) == empty


class TestFileRoundTrip:
    def test_write_run_then_read_report_round_trips(self, tmp_path: Path):
        original = sample_report()
        path = write_run(original, tmp_path)
        assert path == tmp_path / RUNS_DIRNAME / f"{original.config_hash}.json"
        assert read_report(path) == original

    def test_write_run_output_is_pretty_and_stable(self, tmp_path: Path):
        # A file diff should show what changed, not a one-line blob.
        path = write_run(sample_report(), tmp_path)
        text = path.read_text(encoding="utf-8")
        assert text.count("\n") > 5
        assert text.endswith("\n")


class TestBaseline:
    def test_write_then_read_baseline_round_trips_every_run(self, tmp_path: Path):
        runs = {"dense_512": sample_report("dense_512"), "hybrid_512": sample_report("hybrid_512")}
        write_baseline(runs, tmp_path, reason="initial baseline")

        baseline = read_baseline(tmp_path)
        assert set(baseline.runs) == {"dense_512", "hybrid_512"}
        assert baseline.runs["dense_512"] == runs["dense_512"]
        assert baseline.reason == "initial baseline"
        assert baseline.blessed_at

    def test_a_missing_baseline_file_is_a_clear_error(self, tmp_path: Path):
        import pytest

        from groundtruth.scoring.report_io import ReportIOError

        with pytest.raises(ReportIOError, match="baseline"):
            read_baseline(tmp_path)

    def test_reason_may_not_be_blank(self, tmp_path: Path):
        import pytest

        from groundtruth.scoring.report_io import ReportIOError

        with pytest.raises(ReportIOError, match="reason"):
            write_baseline({"dense_512": sample_report()}, tmp_path, reason="   ")
