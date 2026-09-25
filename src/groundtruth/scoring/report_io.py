"""Reading and writing scoring results.

Two files, both committed for auditability:

``results/runs/<config_hash>.json``
    One config's full ``ScoringReport``, written by `gt run`. Addressed by
    content hash, per ADR-0010, so results never collide across configs.

``results/baseline.json``
    The blessed subset the gate compares every run against, written only by
    `gt baseline bless`. A re-bless is its own commit, touching nothing else,
    with a reason -- enforced here by refusing a blank one, because "why" is
    the whole point of the discipline (see how-to-run.md).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from groundtruth.scoring.models import (
    LatencySummary,
    MetricSummary,
    MetricTable,
    ScoringReport,
)

RUNS_DIRNAME: Final[str] = "runs"
BASELINE_FILENAME: Final[str] = "baseline.json"


class ReportIOError(Exception):
    """A result or baseline file could not be read or written."""


def _table_to_dict(table: MetricTable) -> dict[str, dict[str, dict[str, float | int]]]:
    """JSON object keys must be strings; ``k`` (an int) is coerced and restored."""
    return {
        metric: {str(k): asdict(summary) for k, summary in by_k.items()}
        for metric, by_k in table.items()
    }


def _table_from_dict(raw: dict[str, Any]) -> MetricTable:
    return {
        metric: {
            int(k): MetricSummary(
                mean=v["mean"], ci_low=v["ci_low"], ci_high=v["ci_high"], n=v["n"]
            )
            for k, v in by_k.items()
        }
        for metric, by_k in raw.items()
    }


def report_to_dict(report: ScoringReport) -> dict[str, Any]:
    """A JSON-serializable, pretty-printable form of a ``ScoringReport``."""
    return {
        "config_name": report.config_name,
        "config_hash": report.config_hash,
        "golden_set_hash": report.golden_set_hash,
        "corpus_manifest_hash": report.corpus_manifest_hash,
        "code_version": report.code_version,
        "run_fingerprint": report.run_fingerprint,
        "n_queries": report.n_queries,
        "n_skipped": report.n_skipped,
        "overall": _table_to_dict(report.overall),
        "by_category": {name: _table_to_dict(table) for name, table in report.by_category.items()},
        "by_origin": {name: _table_to_dict(table) for name, table in report.by_origin.items()},
        "latency": asdict(report.latency),
    }


def report_from_dict(raw: dict[str, Any]) -> ScoringReport:
    return ScoringReport(
        config_name=raw["config_name"],
        config_hash=raw["config_hash"],
        golden_set_hash=raw["golden_set_hash"],
        corpus_manifest_hash=raw["corpus_manifest_hash"],
        code_version=raw["code_version"],
        run_fingerprint=raw["run_fingerprint"],
        n_queries=raw["n_queries"],
        n_skipped=raw["n_skipped"],
        overall=_table_from_dict(raw["overall"]),
        by_category={name: _table_from_dict(table) for name, table in raw["by_category"].items()},
        by_origin={name: _table_from_dict(table) for name, table in raw["by_origin"].items()},
        latency=LatencySummary(**raw["latency"]),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_run(report: ScoringReport, results_dir: Path) -> Path:
    """Write one config's report, addressed by its content hash."""
    path = results_dir / RUNS_DIRNAME / f"{report.config_hash}.json"
    _write_json(path, report_to_dict(report))
    return path


def read_report(path: Path) -> ScoringReport:
    if not path.is_file():
        raise ReportIOError(f"result file not found: {path}")
    try:
        return report_from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ReportIOError(f"{path.name} is not a valid result file: {exc}") from exc


class Baseline:
    """The blessed set of reports the gate compares every run against."""

    __slots__ = ("blessed_at", "reason", "runs")

    def __init__(self, *, blessed_at: str, reason: str, runs: dict[str, ScoringReport]) -> None:
        self.blessed_at = blessed_at
        self.reason = reason
        self.runs = runs


def write_baseline(runs: dict[str, ScoringReport], results_dir: Path, *, reason: str) -> Path:
    """Bless a set of runs as the baseline. Requires a non-blank reason.

    A re-bless is its own commit, touching nothing else, with a message that
    says why (how-to-run.md). A blank reason would make that discipline
    unenforceable at the one point it can actually be checked.
    """
    if not reason.strip():
        raise ReportIOError("a baseline needs a reason: why is this the new blessed run?")

    path = results_dir / BASELINE_FILENAME
    payload = {
        "blessed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": reason.strip(),
        "runs": {name: report_to_dict(report) for name, report in runs.items()},
    }
    _write_json(path, payload)
    return path


def read_baseline(results_dir: Path) -> Baseline:
    path = results_dir / BASELINE_FILENAME
    if not path.is_file():
        raise ReportIOError(
            f"no baseline found at {path}. Run `gt run --all` then "
            f"`gt baseline bless` before the gate has anything to compare against."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Baseline(
            blessed_at=raw["blessed_at"],
            reason=raw["reason"],
            runs={name: report_from_dict(report) for name, report in raw["runs"].items()},
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ReportIOError(f"{BASELINE_FILENAME} is not a valid baseline file: {exc}") from exc


__all__ = [
    "BASELINE_FILENAME",
    "RUNS_DIRNAME",
    "Baseline",
    "ReportIOError",
    "read_baseline",
    "read_report",
    "report_from_dict",
    "report_to_dict",
    "write_baseline",
    "write_run",
]
