"""Loading configurations from YAML.

The registry is where a human-authored file becomes a validated, hashed
configuration. Its most important job is refusing two configurations that are
materially identical under different names -- otherwise the comparison table
silently reports the same experiment twice.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from groundtruth.config.registry import (
    ConfigError,
    load_all_configs,
    load_config,
    shipped_configs_dir,
)

from .test_config_models import dense_payload, hybrid_payload


def write_config(directory: Path, filename: str, payload: dict) -> Path:
    path = directory / filename
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


class TestLoadConfig:
    def test_loads_a_valid_file(self, tmp_path: Path):
        path = write_config(tmp_path, "dense_512.yaml", dense_payload())
        assert load_config(path).name == "dense_512"

    def test_rejects_a_missing_file(self, tmp_path: Path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_rejects_invalid_yaml(self, tmp_path: Path):
        path = tmp_path / "broken.yaml"
        path.write_text("{{{not yaml", encoding="utf-8")
        with pytest.raises(ConfigError, match="parse"):
            load_config(path)

    def test_rejects_a_non_mapping_document(self, tmp_path: Path):
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="mapping"):
            load_config(path)

    def test_error_names_the_offending_file(self, tmp_path: Path):
        # A validation failure across a directory of configs is useless if it
        # does not say which file is wrong.
        path = write_config(tmp_path, "bad.yaml", dense_payload(top_k=50, dense_top_n=1))
        with pytest.raises(ConfigError, match=r"bad\.yaml"):
            load_config(path)

    def test_filename_must_match_the_declared_name(self, tmp_path: Path):
        # Results are written per config name; a mismatch makes the file on
        # disk and the row in the report disagree about what ran.
        path = write_config(tmp_path, "not_the_name.yaml", dense_payload(name="dense_512"))
        with pytest.raises(ConfigError, match="filename"):
            load_config(path)


class TestLoadAllConfigs:
    def test_loads_every_config_in_a_directory(self, tmp_path: Path):
        write_config(tmp_path, "dense_512.yaml", dense_payload())
        write_config(tmp_path, "hybrid_512.yaml", hybrid_payload())
        configs = load_all_configs(tmp_path)
        assert set(configs) == {"dense_512", "hybrid_512"}

    def test_is_keyed_by_config_name(self, tmp_path: Path):
        write_config(tmp_path, "dense_512.yaml", dense_payload())
        assert load_all_configs(tmp_path)["dense_512"].name == "dense_512"

    def test_rejects_two_configs_with_the_same_content_hash(self, tmp_path: Path):
        # Same settings, different names. Running both wastes time and reports
        # one experiment as two independent data points.
        write_config(tmp_path, "alpha.yaml", dense_payload(name="alpha"))
        write_config(tmp_path, "beta.yaml", dense_payload(name="beta"))
        with pytest.raises(ConfigError, match="identical"):
            load_all_configs(tmp_path)

    def test_duplicate_hash_error_names_both_configs(self, tmp_path: Path):
        write_config(tmp_path, "alpha.yaml", dense_payload(name="alpha"))
        write_config(tmp_path, "beta.yaml", dense_payload(name="beta"))
        with pytest.raises(ConfigError) as exc:
            load_all_configs(tmp_path)
        assert "alpha" in str(exc.value) and "beta" in str(exc.value)

    def test_rejects_a_missing_directory(self, tmp_path: Path):
        with pytest.raises(ConfigError, match="not found"):
            load_all_configs(tmp_path / "nope")

    def test_rejects_an_empty_directory(self, tmp_path: Path):
        # Silently returning {} would make the whole eval a no-op that "passes".
        with pytest.raises(ConfigError, match="no configuration"):
            load_all_configs(tmp_path)

    def test_ignores_non_yaml_files(self, tmp_path: Path):
        write_config(tmp_path, "dense_512.yaml", dense_payload())
        (tmp_path / "README.md").write_text("notes", encoding="utf-8")
        assert set(load_all_configs(tmp_path)) == {"dense_512"}

    def test_gate_policy_is_not_treated_as_a_retrieval_config(self, tmp_path: Path):
        # gate_policy.yaml lives in the same directory but is a different kind
        # of document. Loading it as a RetrievalConfig must not be attempted.
        write_config(tmp_path, "dense_512.yaml", dense_payload())
        (tmp_path / "gate_policy.yaml").write_text(
            "primary_metric: recall_at_10\n", encoding="utf-8"
        )
        assert set(load_all_configs(tmp_path)) == {"dense_512"}


class TestShippedConfigs:
    """The four configs this project actually compares."""

    def test_directory_exists(self):
        assert shipped_configs_dir().is_dir()

    def test_all_shipped_configs_are_valid(self):
        configs = load_all_configs(shipped_configs_dir())
        assert set(configs) == {"dense_512", "dense_256", "hybrid_512", "hybrid_512_rerank"}

    def test_every_shipped_config_has_a_distinct_hash(self):
        configs = load_all_configs(shipped_configs_dir())
        hashes = {name: cfg.config_hash for name, cfg in configs.items()}
        assert len(set(hashes.values())) == len(hashes)

    def test_the_two_chunk_size_arms_differ_only_in_chunking(self):
        # The chunk-size comparison is only meaningful if nothing else moved.
        configs = load_all_configs(shipped_configs_dir())
        a = configs["dense_512"].model_dump(mode="json", exclude={"name", "description"})
        b = configs["dense_256"].model_dump(mode="json", exclude={"name", "description"})
        differing = {k for k in a if a[k] != b[k]}
        assert differing == {"chunking"}

    def test_only_one_shipped_config_enables_the_reranker(self):
        configs = load_all_configs(shipped_configs_dir())
        enabled = {n for n, c in configs.items() if c.reranker.enabled}
        assert enabled == {"hybrid_512_rerank"}
