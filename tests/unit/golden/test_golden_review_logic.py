"""Review logic: the queue, the edit round-trip, and what a decision records.

Two hours of human judgment is the project's critical path (ADR-0009). A bug
in "which candidate comes next" or "what did the reviewer actually change"
costs some of it and is invisible at the time, so all of it is tested without
a terminal in the loop.
"""

from __future__ import annotations

import pytest

from groundtruth.golden.candidates import Candidate, query_id_for
from groundtruth.golden.review import (
    ReviewContext,
    ReviewError,
    accept,
    echo_warning,
    edited_fields,
    from_editable,
    pending,
    reject,
    render_candidate,
    skip,
    source_passage,
    to_editable,
)
from groundtruth.golden.models import RelevanceLabel
from groundtruth.golden.overlap import echo_signal

HOLDING = "Summary judgment is appropriate only where no genuine dispute exists."
OTHER = "The nonmoving party must set forth specific facts showing an issue."
PREFIX = "Before the analysis. "
DOCUMENT = "x" * 5000 + PREFIX + HOLDING + " " + OTHER + " After, at length."
WINDOW_START = 5000
WINDOW_END = len(DOCUMENT)

CONTEXT = ReviewContext(reviewer="rahul", decided_at="2026-09-13T10:00:00+00:00")


def label(quote: str = HOLDING, gain: int = 3) -> RelevanceLabel:
    start = DOCUMENT.index(quote)
    return RelevanceLabel(
        doc_id="cl-1", char_start=start, char_end=start + len(quote), gain=gain, quote=quote
    )


def candidate(candidate_id: str = "c-0001", **kwargs: object) -> Candidate:
    fields: dict[str, object] = {
        "candidate_id": candidate_id,
        "query": "when may a court decide a case without a trial",
        "category": "factual-lookup",
        "labels": (label(),),
        "generator_model": "claude-sonnet-5-20260101",
        "prompt_version": "golden-candidate-v1",
        "source_doc_id": "cl-1",
        "source_char_start": WINDOW_START,
        "source_char_end": WINDOW_END,
    }
    fields.update(kwargs)
    return Candidate(**fields)  # type: ignore[arg-type]


def passage_of(cand: Candidate = None):  # type: ignore[assignment]
    return source_passage(cand or candidate(), DOCUMENT)


class TestQueryIdFor:
    def test_swaps_the_candidate_prefix(self):
        assert query_id_for("c-0042") == "q-0042"

    def test_is_stable_for_an_unprefixed_id(self):
        assert query_id_for("abc") == "q-abc"

    def test_is_a_pure_function_of_the_candidate(self):
        # Not a counter: reversing an earlier decision must not renumber the
        # set and silently change the golden-set hash.
        assert query_id_for("c-0042") == query_id_for("c-0042")


class TestPending:
    def test_everything_is_pending_before_any_review(self):
        candidates = (candidate("c-1"), candidate("c-2"))
        assert pending(candidates, []) == candidates

    def test_a_settled_candidate_drops_out(self):
        candidates = (candidate("c-1"), candidate("c-2"))
        decided = accept(candidates[0], CONTEXT)
        assert [c.candidate_id for c in pending(candidates, [decided])] == ["c-2"]

    def test_a_rejected_candidate_drops_out(self):
        candidates = (candidate("c-1"),)
        assert pending(candidates, [reject(candidates[0], CONTEXT, "verbatim echo")]) == ()

    def test_a_skipped_candidate_comes_back(self):
        # "Skip" means come back to this. A queue that dropped skips would
        # turn indecision into silent exclusion from the golden set.
        candidates = (candidate("c-1"),)
        assert pending(candidates, [skip(candidates[0], CONTEXT)]) == candidates

    def test_skips_can_be_excluded_for_a_final_pass(self):
        candidates = (candidate("c-1"),)
        assert pending(candidates, [skip(candidates[0], CONTEXT)], include_skipped=False) == ()

    def test_a_reversal_puts_a_candidate_back_in_the_queue(self):
        candidates = (candidate("c-1"),)
        log = [accept(candidates[0], CONTEXT), skip(candidates[0], CONTEXT)]
        assert pending(candidates, log) == candidates

    def test_preserves_candidate_order(self):
        candidates = tuple(candidate(f"c-{i}") for i in range(5))
        assert pending(candidates, []) == candidates


class TestRendering:
    def test_marks_the_labeled_span_inside_the_passage(self):
        # The reviewer has to see the span in context: whether a quote is the
        # holding or a fragment of one is not visible from the quote alone.
        out = render_candidate(candidate(), passage_of(), 1, 10)
        assert f"[[gain 3>>{HOLDING}<<]]" in out
        assert PREFIX.strip() in out

    def test_shows_position_category_and_echo(self):
        out = render_candidate(candidate(), passage_of(), 3, 10)
        assert "[3/10]" in out
        assert "factual-lookup" in out
        assert "ECHO" in out

    def test_marks_several_spans(self):
        cand = candidate(labels=(label(), label(OTHER, gain=2)))
        out = render_candidate(cand, passage_of(cand), 1, 1)
        assert "[[gain 3>>" in out
        assert "[[gain 2>>" in out


class TestEchoWarning:
    def test_silent_on_a_genuine_paraphrase(self):
        assert echo_warning(echo_signal("when may a court skip trial", HOLDING)) == ""

    def test_notes_a_verbatim_lift(self):
        warning = echo_warning(echo_signal(HOLDING, HOLDING))
        assert "possible echo" in warning
        assert "consecutive tokens" in warning

    def test_notes_rearranged_vocabulary(self):
        warning = echo_warning(echo_signal("dispute genuine judgment summary", HOLDING))
        assert "query tokens appear in the passage" in warning


class TestEditRoundTrip:
    def test_an_unedited_round_trip_changes_nothing(self):
        cand = candidate()
        assert from_editable(to_editable(cand), cand, passage_of(cand)) == cand

    def test_the_editable_form_explains_the_gain_scale(self):
        text = to_editable(candidate())
        assert "dispositive" in text
        assert "factual-lookup" in text

    def test_editing_the_query_is_detected(self):
        cand = candidate()
        edited = from_editable(
            to_editable(cand).replace(cand.query, "a different question entirely"),
            cand,
            passage_of(cand),
        )
        assert edited_fields(cand, edited) == ("query",)

    def test_editing_a_quote_re_derives_its_offsets(self):
        # Offsets must follow the quote. A reviewer narrowing a span and
        # leaving stale offsets behind is exactly how a label ends up pointing
        # at the wrong characters.
        cand = candidate()
        edited = from_editable(to_editable(cand).replace(HOLDING, OTHER), cand, passage_of(cand))
        assert edited.labels[0].quote == OTHER
        assert edited.labels[0].char_start == DOCUMENT.index(OTHER)
        assert edited_fields(cand, edited) == ("labels",)

    def test_a_quote_that_is_no_longer_in_the_passage_is_refused(self):
        cand = candidate()
        with pytest.raises(ReviewError, match="quote-not-found"):
            from_editable(
                to_editable(cand).replace(HOLDING, "A sentence that is simply not there."),
                cand,
                passage_of(cand),
            )

    def test_an_unknown_category_is_refused(self):
        cand = candidate()
        with pytest.raises(ReviewError, match="unknown category"):
            from_editable(
                to_editable(cand).replace("factual-lookup", "interesting"), cand, passage_of(cand)
            )

    def test_broken_yaml_is_refused_with_a_readable_message(self):
        cand = candidate()
        with pytest.raises(ReviewError, match="valid YAML"):
            from_editable("query: [unclosed", cand, passage_of(cand))

    def test_an_emptied_query_is_refused(self):
        cand = candidate()
        with pytest.raises(ReviewError, match="no query"):
            from_editable("query: ''\ncategory: factual-lookup\nspans: []\n", cand, passage_of(cand))

    def test_removing_every_span_is_refused(self):
        cand = candidate()
        with pytest.raises(ReviewError, match="no spans"):
            from_editable(
                f"query: {cand.query}\ncategory: factual-lookup\nspans: []\n",
                cand,
                passage_of(cand),
            )


class TestDecisions:
    def test_an_acceptance_records_the_generator_and_the_reviewer(self):
        decision = accept(candidate(), CONTEXT, seconds=37.5)
        assert decision.action == "accept"
        assert decision.pair is not None
        provenance = decision.pair.provenance
        assert provenance.origin == "llm"
        assert provenance.generator_model == "claude-sonnet-5-20260101"
        assert provenance.reviewer == "rahul"
        assert provenance.review_seconds == 37.5
        assert provenance.edited_fields == ()

    def test_the_pair_takes_its_query_id_from_the_candidate(self):
        decision = accept(candidate("c-0007"), CONTEXT)
        assert decision.pair is not None
        assert decision.pair.query_id == "q-0007"

    def test_an_edit_records_what_changed(self):
        cand = candidate()
        revised = cand.model_copy(update={"query": "a materially different question"})
        decision = accept(cand, CONTEXT, revised=revised)
        assert decision.action == "edit"
        assert decision.pair is not None
        assert decision.pair.provenance.review_action == "edited"
        assert decision.pair.provenance.edited_fields == ("query",)

    def test_the_changed_fields_are_derived_not_declared(self):
        # Passing an identical "revision" is an acceptance, not an edit. The
        # record cannot disagree with what was actually changed.
        cand = candidate()
        decision = accept(cand, CONTEXT, revised=cand)
        assert decision.action == "accept"
        assert decision.pair is not None
        assert decision.pair.provenance.edited_fields == ()

    def test_a_rejection_keeps_its_reason(self):
        decision = reject(candidate(), CONTEXT, "  echoes the passage verbatim  ")
        assert decision.action == "reject"
        assert decision.reason == "echoes the passage verbatim"
        assert decision.pair is None

    def test_a_rejection_without_a_reason_is_refused(self):
        with pytest.raises(ReviewError, match="reason"):
            reject(candidate(), CONTEXT, "   ")

    def test_a_skip_carries_no_pair_and_no_reason(self):
        decision = skip(candidate(), CONTEXT)
        assert decision.action == "skip"
        assert decision.pair is None

    def test_accepting_a_candidate_with_no_relevant_span_fails_loudly(self):
        # GoldenPair requires gain >= 2. Accepting a marginal proposal
        # unchanged is a reviewer error, and it surfaces here rather than
        # producing a pair that every metric would silently skip.
        marginal = candidate(labels=(label(gain=1),))
        with pytest.raises(Exception, match="gain"):
            accept(marginal, CONTEXT)
