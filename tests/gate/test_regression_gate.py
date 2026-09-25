"""The regression gate itself.

Per ADR-0006, this runs LOCALLY rather than in CI: `pytest -m gate`, or
`make gate`. It is never excluded by default -- see the `gate` marker's own
description in pyproject.toml.

Needs no model, no network, no GPU: every string any config could possibly
need is already in the committed embedding cache (ADR-0003), served through
CachedOnlyEmbedder, which raises loudly on a miss rather than computing one.

Compares the current in-process retrieval path against `results/baseline.json`
using the thresholds in `configs/gate_policy.yaml`. The comparison itself is
`groundtruth.gate.compare.evaluate_gate` -- a pure function, unit-tested on
its own in `tests/unit/gate/test_compare.py`. This file's only job is the
wiring: build every gated config's retriever for real, score it against the
real golden set, and assert on what the pure function says.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from groundtruth import __version__
from groundtruth.config.registry import load_all_configs, shipped_configs_dir
from groundtruth.corpus.snapshot import read_manifest, verify_snapshot
from groundtruth.embedding.cache import load_cached_only_embedder
from groundtruth.gate.compare import ConfigGateResult, evaluate_gate
from groundtruth.gate.policy import load_gate_policy
from groundtruth.golden.store import read_golden_set
from groundtruth.paths import cache_dir, configs_dir, corpus_dir, golden_dir
from groundtruth.retrieval.build import build_retriever
from groundtruth.scoring.evaluate import score_config
from groundtruth.scoring.models import ScoringReport
from groundtruth.scoring.report_io import ReportIOError, read_baseline

pytestmark = pytest.mark.gate

#: Loaded at collection time so the parametrize list below is fixed thresholds
#: decided before any number exists to be tempted by, per the policy file's
#: own header comment -- not something the fixtures below can influence.
_POLICY = load_gate_policy(configs_dir() / "gate_policy.yaml")


def _current_reports() -> dict[str, ScoringReport]:
    """Score every implemented, non-reranker config against the golden set."""
    documents = verify_snapshot(corpus_dir())
    manifest_hash = read_manifest(corpus_dir()).manifest_hash
    golden = read_golden_set(golden_dir())
    configs = load_all_configs(shipped_configs_dir())

    reports: dict[str, ScoringReport] = {}
    for name, config in configs.items():
        if config.reranker.enabled:
            # Retriever itself refuses this (Phase 9); nothing to score yet.
            continue
        embedder = load_cached_only_embedder(
            config.embedding.model_id, config.embedding.revision, cache_dir() / "embeddings"
        )
        retriever = build_retriever(config, documents, embedder)
        reports[name] = score_config(
            retriever, golden, corpus_manifest_hash=manifest_hash, code_version=__version__
        )
    return reports


@pytest.fixture(scope="module")
def gate_results() -> dict[str, ConfigGateResult]:
    try:
        baseline = read_baseline(Path("results"))
    except ReportIOError as exc:
        pytest.skip(f"no baseline to gate against: {exc}")

    current = _current_reports()
    results = evaluate_gate(baseline.runs, current, _POLICY)
    return {result.config_name: result for result in results}


def _metric_cases() -> list[tuple[str, str]]:
    return [
        (config_name, metric_key)
        for config_name in _POLICY.gated_configs
        for metric_key in sorted(_POLICY.max_absolute_drop)
    ]


@pytest.mark.parametrize("config_name", _POLICY.gated_configs)
def test_identity_matches_the_baseline(
    config_name: str, gate_results: dict[str, ConfigGateResult]
) -> None:
    """The anti-cheat: golden set, corpus, and (if configured) config identity."""
    result = gate_results[config_name]
    if result.status == "skipped_no_baseline":
        pytest.skip(f"{config_name}: no baseline yet")
    assert not result.identity_failures, "\n".join(result.identity_failures)


@pytest.mark.parametrize(("config_name", "metric_key"), _metric_cases())
def test_no_metric_regression(
    config_name: str, metric_key: str, gate_results: dict[str, ConfigGateResult]
) -> None:
    result = gate_results[config_name]
    if result.status == "skipped_no_baseline":
        pytest.skip(f"{config_name}: no baseline yet")
    if metric_key in result.unreported_metrics:
        pytest.fail(
            f"{config_name}: {metric_key} is not reported by both the baseline and this run"
        )

    matching = [
        regression
        for regression in result.regressions
        if f"{regression.metric}_at_{regression.k}" == metric_key
    ]
    assert not matching, str(matching[0])
