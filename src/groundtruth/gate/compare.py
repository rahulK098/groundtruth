"""Comparing a baseline against a current run, under a policy.

Pure: (baseline runs, current runs, policy) -> a verdict per gated config.
Nothing here reads a file or builds a retriever -- that wiring lives in
`tests/gate/test_regression_gate.py`, which is the one place this module's
output is actually asserted on.

The asymmetry that matters (ADR-0010): a golden-set or corpus change is a
hard failure, because either one moving underneath a baseline is how a
regression gets "fixed" by deleting the queries it fails. A config change is
the workflow, so it only warns.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from groundtruth.gate.policy import GatePolicy
from groundtruth.scoring.models import ScoringReport

Status = Literal["passed", "failed", "skipped_no_baseline"]

#: A drop exactly at the policy's limit must pass -- "tolerated before the
#: gate fails" (gate_policy.yaml) means <=, not <. Two means-of-90-queries
#: subtracted in float64 do not reproduce a decimal limit exactly (0.850 -
#: 0.840 is 0.010000000000000009), so the comparison needs slack far smaller
#: than any tolerance this project would ever configure.
_EPSILON: Final[float] = 1e-9


@dataclass(frozen=True, slots=True)
class MetricRegression:
    """One (config, metric, k) whose drop exceeded the policy's tolerance."""

    config_name: str
    metric: str
    k: int
    baseline_mean: float
    current_mean: float
    limit: float

    @property
    def drop(self) -> float:
        """Positive means the metric fell; negative means it improved."""
        return self.baseline_mean - self.current_mean

    def __str__(self) -> str:
        delta = self.current_mean - self.baseline_mean
        return (
            f"{self.config_name} {self.metric}@{self.k}: "
            f"{self.baseline_mean:.3f} -> {self.current_mean:.3f} "
            f"({delta:+.3f}, limit -{self.limit:.3f})"
        )


@dataclass(frozen=True, slots=True)
class ConfigGateResult:
    """One config's verdict: whether it passed, and why not if it did not."""

    config_name: str
    status: Status
    #: Golden set / corpus / config hash mismatches, and "measured before but
    #: missing now" -- the anti-cheat checks, independent of any one metric.
    identity_failures: tuple[str, ...] = ()
    #: A metric the policy gates but that is not reported by both runs.
    unreported_metrics: tuple[str, ...] = ()
    regressions: tuple[MetricRegression, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status != "failed"


def _parse_metric_key(metric_key: str) -> tuple[str, int]:
    """ "recall_at_10" -> ("recall", 10)."""
    metric, _, k = metric_key.rpartition("_at_")
    return metric, int(k)


def _identity_failures(
    config_name: str, baseline: ScoringReport, current: ScoringReport, policy: GatePolicy
) -> list[str]:
    """The anti-cheat checks: golden set, corpus, and (if the policy says so) config."""
    failures: list[str] = []

    if policy.require_golden_set_hash_match and current.golden_set_hash != baseline.golden_set_hash:
        failures.append(
            f"{config_name}: golden set changed "
            f"({baseline.golden_set_hash} -> {current.golden_set_hash}); re-bless deliberately"
        )
    if (
        policy.require_corpus_hash_match
        and current.corpus_manifest_hash != baseline.corpus_manifest_hash
    ):
        failures.append(
            f"{config_name}: corpus changed "
            f"({baseline.corpus_manifest_hash} -> {current.corpus_manifest_hash}); re-bless deliberately"
        )
    if current.config_hash != baseline.config_hash and policy.config_hash_mismatch == "fail":
        failures.append(
            f"{config_name}: config changed ({baseline.config_hash} -> {current.config_hash})"
        )
    return failures


def _metric_checks(
    config_name: str, baseline: ScoringReport, current: ScoringReport, policy: GatePolicy
) -> tuple[list[str], list[MetricRegression]]:
    """Every gated metric: reported by both runs, and within tolerance."""
    unreported: list[str] = []
    regressions: list[MetricRegression] = []

    for metric_key, limit in policy.max_absolute_drop.items():
        metric, k = _parse_metric_key(metric_key)
        baseline_summary = baseline.overall.get(metric, {}).get(k)
        current_summary = current.overall.get(metric, {}).get(k)
        if baseline_summary is None or current_summary is None:
            unreported.append(metric_key)
            continue

        regression = MetricRegression(
            config_name=config_name,
            metric=metric,
            k=k,
            baseline_mean=baseline_summary.mean,
            current_mean=current_summary.mean,
            limit=limit,
        )
        if regression.drop > limit + _EPSILON:
            regressions.append(regression)

    return unreported, regressions


def evaluate_config(
    config_name: str,
    baseline_runs: Mapping[str, ScoringReport],
    current_runs: Mapping[str, ScoringReport],
    policy: GatePolicy,
) -> ConfigGateResult:
    """Compare one config's current run against its baseline, if it has one."""
    if config_name not in baseline_runs:
        # Nothing to compare against -- a config built after the baseline
        # was frozen (or, right now, the not-yet-implemented reranker) has
        # no history to regress from.
        return ConfigGateResult(config_name, status="skipped_no_baseline")

    baseline = baseline_runs[config_name]

    if config_name not in current_runs:
        return ConfigGateResult(
            config_name,
            status="failed",
            identity_failures=(
                f"{config_name}: was measured in the baseline but is missing from this run",
            ),
        )

    current = current_runs[config_name]
    identity_failures = _identity_failures(config_name, baseline, current, policy)
    unreported, regressions = _metric_checks(config_name, baseline, current, policy)

    status: Status = "failed" if (identity_failures or unreported or regressions) else "passed"
    return ConfigGateResult(
        config_name,
        status=status,
        identity_failures=tuple(identity_failures),
        unreported_metrics=tuple(unreported),
        regressions=tuple(regressions),
    )


def evaluate_gate(
    baseline_runs: Mapping[str, ScoringReport],
    current_runs: Mapping[str, ScoringReport],
    policy: GatePolicy,
) -> tuple[ConfigGateResult, ...]:
    """One verdict per config the policy gates, in the policy's own order."""
    return tuple(
        evaluate_config(name, baseline_runs, current_runs, policy) for name in policy.gated_configs
    )
