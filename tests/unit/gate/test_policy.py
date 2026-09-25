"""Loading configs/gate_policy.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from groundtruth.gate.policy import GatePolicy, GatePolicyError, load_gate_policy

VALID = {
    "primary_metric": "recall_at_10",
    "gated_configs": ["dense_512", "hybrid_512"],
    "max_absolute_drop": {"recall_at_10": 0.01, "ndcg_at_10": 0.015},
    "require_golden_set_hash_match": True,
    "require_corpus_hash_match": True,
    "config_hash_mismatch": "warn",
    "per_category": "report_only",
}


def write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "gate_policy.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


class TestLoadGatePolicy:
    def test_loads_a_valid_policy(self, tmp_path: Path):
        policy = load_gate_policy(write(tmp_path, VALID))
        assert policy.primary_metric == "recall_at_10"
        assert policy.gated_configs == ("dense_512", "hybrid_512")
        assert policy.max_absolute_drop == {"recall_at_10": 0.01, "ndcg_at_10": 0.015}

    def test_the_shipped_policy_file_loads(self):
        # The real committed file, not a fixture -- catches drift between the
        # model and what is actually on disk.
        policy = load_gate_policy(Path("configs/gate_policy.yaml"))
        assert policy.primary_metric
        assert policy.gated_configs

    def test_missing_file_is_a_clear_error(self, tmp_path: Path):
        with pytest.raises(GatePolicyError, match="not found"):
            load_gate_policy(tmp_path / "nope.yaml")

    def test_malformed_yaml_is_a_clear_error(self, tmp_path: Path):
        path = tmp_path / "gate_policy.yaml"
        path.write_text("not: [valid", encoding="utf-8")
        with pytest.raises(GatePolicyError):
            load_gate_policy(path)

    def test_an_invalid_config_hash_mismatch_value_is_refused(self, tmp_path: Path):
        bad = {**VALID, "config_hash_mismatch": "ignore"}
        with pytest.raises(GatePolicyError):
            load_gate_policy(write(tmp_path, bad))

    def test_defaults_are_the_conservative_choice(self, tmp_path: Path):
        minimal = {
            "primary_metric": "recall_at_10",
            "gated_configs": ["dense_512"],
            "max_absolute_drop": {"recall_at_10": 0.01},
        }
        policy = load_gate_policy(write(tmp_path, minimal))
        assert policy.require_golden_set_hash_match is True
        assert policy.require_corpus_hash_match is True
        assert policy.config_hash_mismatch == "warn"


class TestGatePolicyIsFrozen:
    def test_cannot_be_mutated(self):
        policy = GatePolicy(
            primary_metric="recall_at_10",
            gated_configs=("dense_512",),
            max_absolute_drop={"recall_at_10": 0.01},
        )
        with pytest.raises(Exception):  # noqa: B017 -- pydantic's own FrozenInstanceError-equivalent
            policy.primary_metric = "ndcg_at_10"  # type: ignore[misc]
