"""``gt golden`` surface.

The review loop is the project's critical path, so beyond the usual "each
command is reachable and fails loudly" these tests pin the properties that
protect two hours of human work: every keystroke lands in the log before the
next candidate is shown, quitting mid-review loses nothing, and the golden set
on disk always matches the log.

`golden generate` is not exercised end to end -- it needs a key and a network.
Its logic lives in `golden/proposals.py` and is tested there.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from groundtruth.cli import golden as golden_commands
from groundtruth.cli.main import app
from groundtruth.corpus.snapshot import write_snapshot
from groundtruth.golden.candidates import Candidate
from groundtruth.golden.models import RelevanceLabel
from groundtruth.golden.store import (
    CANDIDATES_FILENAME,
    GOLDEN_SET_FILENAME,
    REVIEW_LOG_FILENAME,
    append_candidates,
    read_candidates,
    read_golden_set,
    read_review_log,
)
from groundtruth.llm.models import Completion, ProviderRequestError
from tests.fixtures.mini_corpus import mini_documents

runner = CliRunner()

QUOTE = (
    "Summary judgment is appropriate only where there is no\n"
    "genuine dispute as to any material fact"
)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    directory = tmp_path / "corpus"
    write_snapshot(mini_documents(), directory, corpus_id="mini", source="test")
    return directory


@pytest.fixture
def golden(tmp_path: Path) -> Path:
    return tmp_path / "golden"


def candidate(candidate_id: str, query: str) -> Candidate:
    text = mini_documents()[0].text
    start = text.index(QUOTE)
    return Candidate(
        candidate_id=candidate_id,
        query=query,
        category="factual-lookup",
        labels=(
            RelevanceLabel(
                doc_id="mini-001",
                char_start=start,
                char_end=start + len(QUOTE),
                gain=3,
                quote=QUOTE,
            ),
        ),
        generator_model="claude-sonnet-5-20260101",
        prompt_version="golden-candidate-v1",
        source_doc_id="mini-001",
        source_char_start=0,
        source_char_end=len(text),
    )


def seed_candidates(directory: Path, count: int = 3) -> None:
    append_candidates(
        [
            candidate(f"c-{n:04d}", f"question number {n} about summary judgment")
            for n in range(1, count + 1)
        ],
        directory,
    )


def common(golden: Path, corpus: Path) -> list[str]:
    return ["--directory", str(golden), "--corpus", str(corpus)]


class TestHelp:
    def test_group_is_registered(self):
        result = runner.invoke(app, ["golden", "--help"])
        assert result.exit_code == 0
        for command in ("generate", "add", "review", "rebuild", "validate", "status"):
            assert command in result.output


class TestGenerateCredentials:
    @pytest.mark.parametrize(
        "provider, variable",
        [
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("azure", "AZURE_OPENAI_API_KEY"),
            ("groq", "GROQ_API_KEY"),
        ],
    )
    def test_missing_key_exits_non_zero_naming_the_variable(
        self,
        tmp_path: Path,
        monkeypatch,
        corpus: Path,
        golden: Path,
        provider: str,
        variable: str,
    ):
        monkeypatch.chdir(tmp_path)
        for name in (
            "ANTHROPIC_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_KEY",
            "GROQ_API_KEY",
        ):
            monkeypatch.delenv(name, raising=False)

        result = runner.invoke(
            app,
            ["golden", "generate", "--count", "1", "--provider", provider, *common(golden, corpus)],
        )
        assert result.exit_code == 1
        assert variable in result.output
        assert not (golden / CANDIDATES_FILENAME).exists()

    def test_an_unknown_provider_lists_the_known_ones(
        self, tmp_path: Path, monkeypatch, corpus: Path, golden: Path
    ):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            ["golden", "generate", "--provider", "nope", *common(golden, corpus)],
        )
        assert result.exit_code == 1
        assert "azure" in result.output


@pytest.fixture
def samplable_corpus(tmp_path: Path) -> Path:
    """A corpus whose documents clear the sampler's minimum length.

    The mini opinions are ~1.2k characters, below the 2,000 the sampler
    requires before it will treat a document as carrying a holding worth
    asking about, so they cannot drive `generate` at all.
    """
    directory = tmp_path / "big-corpus"
    documents = tuple(
        # Padded with DISTINCT filler, never by repeating the opinion: a
        # doubled document makes every quote in it ambiguous, which is a
        # property of the fixture rather than of anything under test.
        doc.model_copy(
            update={
                "text": doc.text
                + "\n\n"
                + "\n".join(
                    f"Paragraph {n} of the appendix to {doc.doc_id}, which is filler." * 3
                    for n in range(40)
                )
            }
        )
        for doc in mini_documents()
    )
    write_snapshot(documents, directory, corpus_id="big", source="test")
    return directory


class TestGenerateFailureIsActionable:
    """A run where every call failed must say WHY, and must not block the retry.

    Both properties were missing on the first Azure attempt: 120 failures
    reported as a bare count, and an empty candidates.jsonl left behind that
    then refused the corrected re-run.
    """

    def stub(self, monkeypatch, exc: Exception) -> None:
        class FailingProvider:
            name = "azure"
            model = "chat"

            def complete(self, **_: object) -> object:
                raise exc

        monkeypatch.setattr(golden_commands, "build_provider", lambda *a, **k: FailingProvider())

    def test_the_underlying_error_is_printed_not_just_counted(
        self, monkeypatch, samplable_corpus: Path, golden: Path
    ):
        self.stub(monkeypatch, ProviderRequestError("azure: HTTP 404: DeploymentNotFound"))

        result = runner.invoke(
            app, ["golden", "generate", "--count", "2", *common(golden, samplable_corpus)]
        )
        assert "request-failed" in result.output
        assert "DeploymentNotFound" in result.output

    def test_a_run_that_produced_nothing_leaves_no_file_to_block_the_retry(
        self, monkeypatch, samplable_corpus: Path, golden: Path
    ):
        self.stub(monkeypatch, ProviderRequestError("azure: HTTP 401: denied"))

        result = runner.invoke(
            app, ["golden", "generate", "--count", "2", *common(golden, samplable_corpus)]
        )
        assert result.exit_code == 1
        assert not (golden / CANDIDATES_FILENAME).exists()

    def test_a_partial_run_still_writes_what_it_got(
        self, monkeypatch, samplable_corpus: Path, golden: Path
    ):
        # One failure among several must not discard the successes.
        calls = {"n": 0}

        class FlakyProvider:
            name = "azure"
            model = "chat"

            def complete(self, **kwargs: object) -> Completion:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise ProviderRequestError("azure: HTTP 500: transient")
                passage = str(kwargs["user"]).split("---")[1]
                quote = passage.strip()[:120]
                return Completion(
                    model="gpt-4o-2024-11-20",
                    tool_input={
                        "query": "what standard did the court apply here",
                        "category": "factual-lookup",
                        "spans": [{"quote": quote, "gain": 3}],
                    },
                )

        monkeypatch.setattr(golden_commands, "build_provider", lambda *a, **k: FlakyProvider())

        result = runner.invoke(
            app, ["golden", "generate", "--count", "3", *common(golden, samplable_corpus)]
        )
        assert result.exit_code == 0, result.output
        assert len(read_candidates(golden)) == 2
        assert "request-failed" in result.output


class TestGenerateGuards:
    def test_refuses_to_overwrite_existing_candidates(
        self, monkeypatch, corpus: Path, golden: Path
    ):
        # Candidate ids are positional, so a second run would collide with the
        # first and the review log would point at the wrong proposals.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
        seed_candidates(golden)

        result = runner.invoke(app, ["golden", "generate", "--count", "1", *common(golden, corpus)])
        assert result.exit_code == 1
        assert CANDIDATES_FILENAME in result.output


class TestReview:
    def test_accept_reject_skip_each_land_in_the_log(self, corpus: Path, golden: Path):
        seed_candidates(golden, 3)

        result = runner.invoke(
            app,
            ["golden", "review", "--reviewer", "rahul", *common(golden, corpus)],
            input="a\nr\nnot a real question\ns\n",
        )
        assert result.exit_code == 0, result.output

        log = read_review_log(golden)
        assert [d.action for d in log] == ["accept", "reject", "skip"]
        assert log[1].reason == "not a real question"
        assert all(d.reviewer == "rahul" for d in log)

        # The set on disk is materialized from the log at exit.
        assert [p.query_id for p in read_golden_set(golden).pairs] == ["q-0001"]

    def test_quit_mid_review_keeps_every_decision_so_far(self, corpus: Path, golden: Path):
        seed_candidates(golden, 3)

        result = runner.invoke(
            app,
            ["golden", "review", "--reviewer", "rahul", *common(golden, corpus)],
            input="a\nq\n",
        )
        assert result.exit_code == 0, result.output
        assert [d.action for d in read_review_log(golden)] == ["accept"]
        assert len(read_golden_set(golden).pairs) == 1

    def test_resumes_with_only_the_undecided_candidates(self, corpus: Path, golden: Path):
        seed_candidates(golden, 3)
        runner.invoke(
            app,
            ["golden", "review", "--reviewer", "rahul", *common(golden, corpus)],
            input="a\nq\n",
        )

        result = runner.invoke(
            app,
            ["golden", "review", "--reviewer", "rahul", *common(golden, corpus)],
            input="q\n",
        )
        assert "c-0001" not in result.output
        assert "c-0002" in result.output

    def test_rejection_without_a_reason_is_asked_again_not_recorded(
        self, corpus: Path, golden: Path
    ):
        seed_candidates(golden, 1)

        result = runner.invoke(
            app,
            ["golden", "review", "--reviewer", "rahul", *common(golden, corpus)],
            input="r\n\nq\n",
        )
        assert result.exit_code == 0, result.output
        assert read_review_log(golden) == ()

    def test_edit_records_the_changed_fields(self, monkeypatch, corpus: Path, golden: Path):
        seed_candidates(golden, 1)

        def fake_editor(text: str, **_: object) -> str:
            return text.replace(
                "question number 1 about summary judgment", "what is the summary judgment standard"
            )

        monkeypatch.setattr(golden_commands, "_edit", fake_editor)

        result = runner.invoke(
            app,
            ["golden", "review", "--reviewer", "rahul", *common(golden, corpus)],
            input="e\n",
        )
        assert result.exit_code == 0, result.output

        (decision,) = read_review_log(golden)
        assert decision.action == "edit"
        assert decision.pair is not None
        assert decision.pair.provenance.edited_fields == ("query",)
        assert decision.pair.query == "what is the summary judgment standard"

    def test_edit_that_breaks_a_quote_is_refused_and_reprompted(
        self, monkeypatch, corpus: Path, golden: Path
    ):
        seed_candidates(golden, 1)

        def fake_editor(text: str, **_: object) -> str:
            return text.replace(QUOTE.splitlines()[0], "not in the passage at all")

        monkeypatch.setattr(golden_commands, "_edit", fake_editor)

        result = runner.invoke(
            app,
            ["golden", "review", "--reviewer", "rahul", *common(golden, corpus)],
            input="e\nq\n",
        )
        assert result.exit_code == 0, result.output
        assert "quote-not-found" in result.output
        assert read_review_log(golden) == ()

    def test_nothing_to_review_says_so(self, corpus: Path, golden: Path):
        result = runner.invoke(
            app, ["golden", "review", "--reviewer", "rahul", *common(golden, corpus)]
        )
        assert result.exit_code == 0
        assert "nothing" in result.output.lower()


class TestAdd:
    def test_hand_authored_pair_enters_the_log_as_human_origin(
        self, monkeypatch, corpus: Path, golden: Path
    ):
        def fake_editor(text: str, **_: object) -> str:
            return (
                "query: what standard applies on a motion for summary judgment\n"
                "category: factual-lookup\n"
                "spans:\n"
                "  - doc_id: mini-001\n"
                "    gain: 3\n"
                f"    quote: {json.dumps(QUOTE)}\n"
            )

        monkeypatch.setattr(golden_commands, "_edit", fake_editor)

        result = runner.invoke(
            app, ["golden", "add", "--reviewer", "rahul", *common(golden, corpus)]
        )
        assert result.exit_code == 0, result.output

        (decision,) = read_review_log(golden)
        assert decision.candidate_id is None
        assert decision.pair is not None
        assert decision.pair.query_id == "q-h-001"
        assert decision.pair.provenance.origin == "human"
        assert read_golden_set(golden).pairs[0].query_id == "q-h-001"

    def test_unchanged_template_adds_nothing(self, monkeypatch, corpus: Path, golden: Path):
        # typer.edit returns None when the editor closes without a change.
        monkeypatch.setattr(golden_commands, "_edit", lambda text, **_: None)

        result = runner.invoke(
            app, ["golden", "add", "--reviewer", "rahul", *common(golden, corpus)]
        )
        assert result.exit_code == 0
        assert not (golden / REVIEW_LOG_FILENAME).exists()

    def test_bad_authoring_exits_non_zero_and_records_nothing(
        self, monkeypatch, corpus: Path, golden: Path
    ):
        monkeypatch.setattr(
            golden_commands, "_edit", lambda text, **_: "query: x\ncategory: nope\nspans: []\n"
        )

        result = runner.invoke(
            app, ["golden", "add", "--reviewer", "rahul", *common(golden, corpus)]
        )
        assert result.exit_code == 1
        assert not (golden / REVIEW_LOG_FILENAME).exists()


class TestRebuildValidateStatus:
    def test_rebuild_rewrites_the_set_from_the_log(self, corpus: Path, golden: Path):
        seed_candidates(golden, 2)
        runner.invoke(
            app,
            ["golden", "review", "--reviewer", "rahul", *common(golden, corpus)],
            input="a\na\n",
        )
        (golden / GOLDEN_SET_FILENAME).write_text("", encoding="utf-8")

        result = runner.invoke(app, ["golden", "rebuild", "--directory", str(golden)])
        assert result.exit_code == 0
        assert len(read_golden_set(golden).pairs) == 2

    def test_validate_passes_an_incomplete_set_with_warnings(self, corpus: Path, golden: Path):
        seed_candidates(golden, 1)
        runner.invoke(
            app,
            ["golden", "review", "--reviewer", "rahul", *common(golden, corpus)],
            input="a\n",
        )

        result = runner.invoke(app, ["golden", "validate", *common(golden, corpus)])
        assert result.exit_code == 0, result.output
        assert "WARNING" in result.output
        assert "set-incomplete" in result.output

    def test_validate_fails_on_a_label_the_corpus_no_longer_matches(
        self, corpus: Path, golden: Path
    ):
        seed_candidates(golden, 1)
        runner.invoke(
            app,
            ["golden", "review", "--reviewer", "rahul", *common(golden, corpus)],
            input="a\n",
        )

        # Re-snapshot the corpus with the labeled document altered underneath
        # the label: the offsets still exist but the text there has changed.
        altered = [
            doc.model_copy(
                update={"text": doc.text.replace("Summary judgment", "Summery judgment")}
            )
            if doc.doc_id == "mini-001"
            else doc
            for doc in mini_documents()
        ]
        write_snapshot(altered, corpus, corpus_id="mini", source="test")

        result = runner.invoke(app, ["golden", "validate", *common(golden, corpus)])
        assert result.exit_code == 1
        assert "quote-mismatch" in result.output

    def test_status_reports_the_counts_the_methodology_quotes(self, corpus: Path, golden: Path):
        seed_candidates(golden, 3)
        runner.invoke(
            app,
            ["golden", "review", "--reviewer", "rahul", *common(golden, corpus)],
            input="a\nr\nnope\ns\n",
        )

        result = runner.invoke(app, ["golden", "status", "--directory", str(golden)])
        assert result.exit_code == 0, result.output
        out = result.output
        assert "generated" in out and "3" in out
        assert "accept" in out and "reject" in out and "skip" in out
        assert "factual-lookup" in out
        assert "sha256:" in out
