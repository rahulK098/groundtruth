"""Integrity checks, and the echo signal that backs the review step.

The quote check below is the single most valuable test in the project: it is
what turns "the corpus was re-normalized under the labels" from an unexplained
metric shift months later into a named error the moment anyone runs verify.
"""

from __future__ import annotations

import pytest

from groundtruth.corpus.models import Document
from groundtruth.golden.models import GoldenPair, GoldenSet, Provenance, RelevanceLabel
from groundtruth.golden.overlap import echo_signal
from groundtruth.golden.validation import (
    MIN_PER_CATEGORY,
    errors,
    validate,
    validate_against_corpus,
    validate_policy,
)

TEXT = "The court held that summary judgment was appropriate on this record."
DOCUMENT = Document(doc_id="cl-1", text=TEXT)
# "summary judgment" begins at index 20.
START, END = 20, 36


def human_provenance() -> Provenance:
    return Provenance(
        origin="human",
        reviewer="rahul",
        reviewed_at="2026-09-13T10:00:00+00:00",
        review_action="accepted",
    )


def llm_provenance() -> Provenance:
    return Provenance(
        origin="llm",
        generator_model="claude-sonnet-5-20260101",
        prompt_version="golden-candidate-v1",
        source_candidate_id="c-1",
        reviewer="rahul",
        reviewed_at="2026-09-13T10:00:00+00:00",
        review_action="accepted",
    )


def pair(
    query_id: str = "q-1",
    *,
    doc_id: str = "cl-1",
    start: int = START,
    end: int = END,
    quote: str | None = None,
    category: str = "factual-lookup",
    provenance: Provenance | None = None,
) -> GoldenPair:
    return GoldenPair(
        query_id=query_id,
        query=f"summary judgment standard {query_id}",
        category=category,  # type: ignore[arg-type]
        labels=(
            RelevanceLabel(
                doc_id=doc_id,
                char_start=start,
                char_end=end,
                gain=3,
                quote=TEXT[start:end] if quote is None else quote,
            ),
        ),
        provenance=provenance or human_provenance(),
    )


class TestAgainstCorpus:
    def test_a_correct_label_produces_no_issues(self):
        assert validate_against_corpus(GoldenSet(pairs=(pair(),)), [DOCUMENT]) == ()

    def test_catches_a_label_pointing_at_an_absent_document(self):
        issues = validate_against_corpus(GoldenSet(pairs=(pair(doc_id="cl-999"),)), [DOCUMENT])
        assert [i.code for i in issues] == ["unknown-document"]
        assert issues[0].query_id == "q-1"

    def test_catches_a_span_past_the_end_of_the_document(self):
        long_quote = "x" * 200
        issues = validate_against_corpus(
            GoldenSet(pairs=(pair(start=0, end=200, quote=long_quote),)), [DOCUMENT]
        )
        assert [i.code for i in issues] == ["span-out-of-range"]

    def test_catches_a_quote_that_no_longer_matches_its_span(self):
        # Exactly what corpus re-normalization looks like: the offsets are
        # still in range, they just point somewhere else now.
        issues = validate_against_corpus(
            GoldenSet(pairs=(pair(start=0, end=16, quote="summary judgment"),)), [DOCUMENT]
        )
        assert [i.code for i in issues] == ["quote-mismatch"]
        assert "re-normalized" in issues[0].message

    def test_reports_every_bad_label_not_just_the_first(self):
        # A single reported error would make fixing a drifted corpus an
        # arbitrarily long loop of run-fix-run.
        golden = GoldenSet(pairs=(pair("q-1", doc_id="cl-9"), pair("q-2", doc_id="cl-8")))
        assert len(validate_against_corpus(golden, [DOCUMENT])) == 2

    def test_every_corpus_issue_is_an_error(self):
        golden = GoldenSet(pairs=(pair(doc_id="cl-999"),))
        assert errors(validate_against_corpus(golden, [DOCUMENT]))


class TestPolicy:
    def test_an_incomplete_set_warns_rather_than_fails(self):
        # The set is legitimately incomplete for the entire time anyone is
        # actually running this, so a hard failure would just get ignored.
        issues = validate_policy(GoldenSet(pairs=(pair(),)))
        assert all(issue.severity == "warning" for issue in issues)
        assert "set-incomplete" in {issue.code for issue in issues}

    def test_a_complete_set_raises_no_size_warning(self):
        pairs = tuple(pair(f"q-{i:03d}") for i in range(4))
        issues = validate_policy(GoldenSet(pairs=pairs), target_size=4, min_per_category=1)
        assert "set-incomplete" not in {issue.code for issue in issues}

    def test_warns_for_each_under_populated_category(self):
        issues = validate_policy(GoldenSet(pairs=(pair(),)))
        under = [i for i in issues if i.code == "category-under-populated"]
        # Every category, including the four with nothing in them at all.
        assert len(under) == 5
        assert str(MIN_PER_CATEGORY) in under[0].message

    def test_warns_when_no_query_was_written_by_a_human(self):
        # Without hand-authored queries the methodology's own check on
        # generator bias cannot be computed at all.
        issues = validate_policy(GoldenSet(pairs=(pair(provenance=llm_provenance()),)))
        assert "no-human-authored-queries" in {issue.code for issue in issues}

    def test_no_such_warning_once_a_human_query_exists(self):
        golden = GoldenSet(pairs=(pair("q-1"), pair("q-2", provenance=llm_provenance())))
        assert "no-human-authored-queries" not in {i.code for i in validate_policy(golden)}

    def test_an_empty_set_does_not_warn_about_origins(self):
        assert "no-human-authored-queries" not in {i.code for i in validate_policy(GoldenSet())}


class TestValidate:
    def test_runs_corpus_checks_before_policy_checks(self):
        # A broken label matters more than an incomplete set, and whoever runs
        # this reads the top of the output.
        issues = validate(GoldenSet(pairs=(pair(doc_id="cl-999"),)), [DOCUMENT])
        assert issues[0].severity == "error"

    def test_an_issue_renders_readably(self):
        issue = validate(GoldenSet(pairs=(pair(doc_id="cl-999"),)), [DOCUMENT])[0]
        rendered = str(issue)
        assert "ERROR" in rendered
        assert "q-1" in rendered


class TestEchoSignal:
    def test_a_verbatim_lift_scores_at_the_ceiling(self):
        signal = echo_signal("summary judgment was appropriate", TEXT)
        assert signal.token_overlap == pytest.approx(1.0)
        assert signal.longest_common_run == 4

    def test_a_genuine_paraphrase_scores_low(self):
        signal = echo_signal("when may a trial judge decide a case without a jury", TEXT)
        assert signal.token_overlap < 0.4
        assert signal.longest_common_run <= 1

    def test_shared_vocabulary_in_a_new_order_still_shows_as_overlap(self):
        # The case unigram overlap catches and the run length does not: every
        # word lifted, rearranged enough to look original.
        signal = echo_signal("appropriate judgment summary record", TEXT)
        assert signal.token_overlap == pytest.approx(1.0)
        assert signal.longest_common_run == 1

    def test_a_long_passage_does_not_inflate_the_run_length(self):
        signal = echo_signal("summary judgment", TEXT * 20)
        assert signal.longest_common_run == 2

    def test_is_case_insensitive(self):
        assert echo_signal("SUMMARY JUDGMENT", TEXT) == echo_signal("summary judgment", TEXT)

    def test_reports_how_many_distinct_tokens_the_query_had(self):
        # A 100% overlap on a three-token query is much weaker evidence than
        # the same number on a twenty-token one.
        assert echo_signal("summary judgment the summary", TEXT).query_tokens == 3

    def test_an_empty_query_scores_zero_rather_than_dividing_by_zero(self):
        assert echo_signal("   ", TEXT) == echo_signal("!!!", TEXT)
        assert echo_signal("   ", TEXT).token_overlap == 0.0

    def test_an_empty_passage_shares_nothing(self):
        signal = echo_signal("summary judgment", "")
        assert signal.token_overlap == 0.0
        assert signal.longest_common_run == 0
