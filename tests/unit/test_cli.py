"""CLI surface.

Thin wrappers, so these tests are correspondingly thin: that each command is
reachable, that failures exit non-zero with an actionable message, and that no
command silently succeeds when it did nothing.

`corpus fetch` is not exercised end to end -- it needs a live API and a
credential. Its logic lives in `corpus/selection.py` and is tested there.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from groundtruth import __version__
from groundtruth.cli.main import app
from groundtruth.corpus.models import Document
from groundtruth.corpus.snapshot import SNAPSHOT_FILENAME, write_snapshot

runner = CliRunner()


class TestTopLevel:
    def test_help_lists_the_command_groups(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "corpus" in result.output

    def test_bare_invocation_shows_help_rather_than_failing_silently(self):
        result = runner.invoke(app, [])
        assert "Usage" in result.output

    def test_version_matches_the_package(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert __version__ in result.output


class TestCorpusHelp:
    def test_group_help(self):
        result = runner.invoke(app, ["corpus", "--help"])
        assert result.exit_code == 0
        assert "fetch" in result.output
        assert "verify" in result.output

    def test_fetch_help_documents_the_defaults(self):
        result = runner.invoke(app, ["corpus", "fetch", "--help"])
        assert result.exit_code == 0
        assert "150" in result.output


class TestCorpusFetchCredentials:
    def test_missing_token_exits_non_zero_with_guidance(self, tmp_path: Path, monkeypatch):
        # Isolate from any real .env, so this asserts behaviour rather than
        # whose machine ran it.
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("COURTLISTENER_API_TOKEN", raising=False)

        result = runner.invoke(app, ["corpus", "fetch", "--limit", "1", "--out", str(tmp_path)])
        assert result.exit_code == 1
        assert "COURTLISTENER_API_TOKEN" in result.output
        # Must not leave a half-written snapshot behind.
        assert not (tmp_path / SNAPSHOT_FILENAME).exists()


class TestCorpusVerify:
    def test_accepts_an_intact_snapshot(self, tmp_path: Path):
        write_snapshot(
            (Document(doc_id="cl-1", text="An opinion."),),
            tmp_path,
            corpus_id="t",
            source="test",
        )
        result = runner.invoke(app, ["corpus", "verify", "--directory", str(tmp_path)])
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_missing_snapshot_exits_non_zero(self, tmp_path: Path):
        result = runner.invoke(app, ["corpus", "verify", "--directory", str(tmp_path)])
        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_corrupt_snapshot_exits_non_zero(self, tmp_path: Path):
        write_snapshot(
            (Document(doc_id="cl-1", text="An opinion."),),
            tmp_path,
            corpus_id="t",
            source="test",
        )
        (tmp_path / SNAPSHOT_FILENAME).write_bytes(b"not zstd at all")
        result = runner.invoke(app, ["corpus", "verify", "--directory", str(tmp_path)])
        assert result.exit_code == 1
