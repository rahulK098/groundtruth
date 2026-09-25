"""``gt run`` and ``gt baseline bless``.

Thin wrappers, so these tests are correspondingly thin: argument validation,
loud failure on a missing corpus or golden set, and one true end-to-end run
against the mini corpus fixture (a real committed-style cache, built with the
deterministic HashingEmbedder) to prove the whole chain -- config, corpus,
cache, golden set, scorer, result file -- is wired correctly with no model
and no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from groundtruth.chunking.pipeline import chunk_corpus
from groundtruth.cli.main import app
from groundtruth.corpus.snapshot import write_snapshot
from groundtruth.embedding.hashing import embedding_key
from groundtruth.embedding.store import store_dir, write_store
from groundtruth.golden.models import GoldenPair, GoldenSet, Provenance, RelevanceLabel
from groundtruth.golden.store import write_golden_set
from tests.fixtures.mini_corpus import (
    MINI_MODEL_ID,
    MINI_REVISION,
    HashingEmbedder,
    mini_config,
    mini_documents,
)

runner = CliRunner()
DOCUMENTS = mini_documents()
BY_ID = {doc.doc_id: doc for doc in DOCUMENTS}


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    """A self-contained project layout: corpus, one config, cache, golden set."""
    corpus_dir = tmp_path / "corpus"
    write_snapshot(DOCUMENTS, corpus_dir, corpus_id="mini", source="test")

    config = mini_config(name="mini_dense", retrieval_mode="dense", top_k=5, dense_top_n=20)
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / f"{config.name}.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8"
    )

    needle = "Summary judgment is appropriate only where there is no"
    text = BY_ID["mini-001"].text
    start = text.index(needle)
    golden = GoldenSet(
        pairs=(
            GoldenPair(
                query_id="q-001",
                query="what is the standard for granting summary judgment",
                category="factual-lookup",
                labels=(
                    RelevanceLabel(
                        doc_id="mini-001",
                        char_start=start,
                        char_end=start + len(needle),
                        gain=3,
                        quote=needle,
                    ),
                ),
                provenance=Provenance(
                    origin="human",
                    reviewer="test",
                    reviewed_at="2026-01-01T00:00:00+00:00",
                    review_action="accepted",
                ),
            ),
        )
    )
    golden_dir = tmp_path / "golden"
    write_golden_set(golden, golden_dir)

    # A committed-style cache: every chunk text plus the one golden query,
    # embedded with the same deterministic stand-in the retrieval tests use.
    chunks = chunk_corpus(DOCUMENTS, config.chunking)
    embedder = HashingEmbedder()
    texts = sorted({chunk.text for chunk in chunks} | {golden.pairs[0].query})
    vectors = embedder.embed(texts)
    keys = [
        embedding_key(text, model_id=MINI_MODEL_ID, revision=MINI_REVISION, normalize=True)
        for text in texts
    ]
    cache_root = tmp_path / "cache" / "embeddings"
    write_store(
        store_dir(cache_root, MINI_MODEL_ID, MINI_REVISION),
        keys,
        np.asarray(vectors, dtype=np.float32),
        model_id=MINI_MODEL_ID,
        revision=MINI_REVISION,
        normalize=True,
    )

    results_dir = tmp_path / "results"
    return {
        "corpus": corpus_dir,
        "configs": configs_dir,
        "golden": golden_dir,
        "cache": tmp_path / "cache",
        "results": results_dir,
        "config_name": config.name,
    }


def common(ws: dict[str, Path]) -> list[str]:
    return [
        "--configs",
        str(ws["configs"]),
        "--corpus",
        str(ws["corpus"]),
        "--cache",
        str(ws["cache"]),
        "--golden",
        str(ws["golden"]),
        "--out",
        str(ws["results"]),
    ]


class TestRunArgumentValidation:
    def test_neither_config_nor_all_is_refused(self, workspace: dict[str, Path]):
        result = runner.invoke(app, ["run", *common(workspace)])
        assert result.exit_code == 1
        assert "--config" in result.output or "--all" in result.output

    def test_both_config_and_all_is_refused(self, workspace: dict[str, Path]):
        result = runner.invoke(app, ["run", "--config", "mini_dense", "--all", *common(workspace)])
        assert result.exit_code == 1

    def test_an_unknown_config_name_fails_loudly(self, workspace: dict[str, Path]):
        result = runner.invoke(app, ["run", "--config", "nope", *common(workspace)])
        assert result.exit_code == 1

    def test_a_missing_corpus_fails_loudly(self, tmp_path: Path, workspace: dict[str, Path]):
        result = runner.invoke(
            app,
            [
                "run",
                "--all",
                "--configs",
                str(workspace["configs"]),
                "--corpus",
                str(tmp_path / "nowhere"),
                "--cache",
                str(workspace["cache"]),
                "--golden",
                str(workspace["golden"]),
                "--out",
                str(workspace["results"]),
            ],
        )
        assert result.exit_code == 1
        assert "FAIL" in result.output


class TestRunEndToEnd:
    def test_writes_a_result_file_addressed_by_config_hash(self, workspace: dict[str, Path]):
        from groundtruth.config.registry import load_config

        result = runner.invoke(
            app, ["run", "--config", workspace["config_name"], *common(workspace)]
        )
        assert result.exit_code == 0, result.output

        config = load_config(workspace["configs"] / f"{workspace['config_name']}.yaml")
        path = workspace["results"] / "runs" / f"{config.config_hash}.json"
        assert path.is_file()

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["config_name"] == workspace["config_name"]
        assert payload["n_queries"] == 1
        assert "recall" in payload["overall"]

    def test_all_runs_every_config_in_the_directory(self, workspace: dict[str, Path]):
        result = runner.invoke(app, ["run", "--all", *common(workspace)])
        assert result.exit_code == 0, result.output
        assert (workspace["results"] / "runs").glob("*.json")

    def test_prints_a_one_line_summary_per_config(self, workspace: dict[str, Path]):
        result = runner.invoke(
            app, ["run", "--config", workspace["config_name"], *common(workspace)]
        )
        assert "recall@10" in result.output or "recall@5" in result.output


class TestBaseline:
    def test_bless_with_no_runs_fails_loudly(self, tmp_path: Path):
        result = runner.invoke(
            app, ["baseline", "bless", "--from", str(tmp_path / "results"), "--reason", "x"]
        )
        assert result.exit_code == 1

    def test_a_blank_reason_is_refused(self, workspace: dict[str, Path]):
        runner.invoke(app, ["run", "--all", *common(workspace)])
        result = runner.invoke(
            app, ["baseline", "bless", "--from", str(workspace["results"]), "--reason", "   "]
        )
        assert result.exit_code == 1

    def test_bless_writes_baseline_json_from_the_runs(self, workspace: dict[str, Path]):
        runner.invoke(app, ["run", "--all", *common(workspace)])
        result = runner.invoke(
            app,
            [
                "baseline",
                "bless",
                "--from",
                str(workspace["results"]),
                "--reason",
                "first numbers",
            ],
        )
        assert result.exit_code == 0, result.output

        baseline_path = workspace["results"] / "baseline.json"
        assert baseline_path.is_file()
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        assert payload["reason"] == "first numbers"
        assert workspace["config_name"] in payload["runs"]
