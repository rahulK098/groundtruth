"""Candidates, the append-only review log, and replay.

The property under test throughout: **the log is the source of truth**. A
re-review appends rather than edits, the golden set is always regenerable from
it, and no decision ever disappears.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from groundtruth.golden.candidates import Candidate, ReviewDecision, materialize
from groundtruth.golden.models import GoldenPair, Provenance, RelevanceLabel
from groundtruth.golden.store import (
    REVIEW_LOG_FILENAME,
    GoldenStoreError,
    append_candidates,
    append_decision,
    read_candidates,
    read_golden_set,
    read_review_log,
    rebuild_golden_set,
    review_summary,
    write_golden_set,
)

QUOTE = "Summary judgment is appropriate"


def label(gain: int = 3, start: int = 100) -> RelevanceLabel:
    return RelevanceLabel(
        doc_id="cl-1", char_start=start, char_end=start + len(QUOTE), gain=gain, quote=QUOTE
    )


def candidate(candidate_id: str = "c-001", **kwargs: object) -> Candidate:
    fields: dict[str, object] = {
        "candidate_id": candidate_id,
        "query": f"what is the standard {candidate_id}",
        "category": "factual-lookup",
        "labels": (label(),),
        "generator_model": "claude-sonnet-5-20260101",
        "prompt_version": "golden-candidate-v1",
        "generated_at": "2026-09-13T09:00:00+00:00",
        "source_doc_id": "cl-1",
        "source_char_start": 0,
        "source_char_end": 2000,
    }
    fields.update(kwargs)
    return Candidate(**fields)  # type: ignore[arg-type]


def reviewed_pair(
    candidate_id: str | None = "c-001",
    *,
    query_id: str = "q-001",
    action: str = "accepted",
    gain: int = 3,
    **kwargs: object,
) -> GoldenPair:
    provenance = Provenance(
        origin="llm" if candidate_id else "human",
        generator_model="claude-sonnet-5-20260101" if candidate_id else None,
        prompt_version="golden-candidate-v1" if candidate_id else None,
        source_candidate_id=candidate_id,
        reviewer="rahul",
        reviewed_at="2026-09-13T10:00:00+00:00",
        review_action=action,  # type: ignore[arg-type]
        edited_fields=("query",) if action == "edited" else (),
    )
    fields: dict[str, object] = {
        "query_id": query_id,
        "query": f"standard for {query_id}",
        "category": "factual-lookup",
        "labels": (label(gain=gain),),
        "provenance": provenance,
    }
    fields.update(kwargs)
    return GoldenPair(**fields)  # type: ignore[arg-type]


def decision(
    candidate_id: str | None = "c-001",
    action: str = "accept",
    *,
    query_id: str = "q-001",
    reason: str = "",
    **kwargs: object,
) -> ReviewDecision:
    produces_pair = action in {"accept", "edit"}
    fields: dict[str, object] = {
        "candidate_id": candidate_id,
        "action": action,
        "reviewer": "rahul",
        "decided_at": "2026-09-13T10:00:00+00:00",
        "review_seconds": 41.0,
        "reason": reason,
        "pair": reviewed_pair(
            candidate_id,
            query_id=query_id,
            action="accepted" if action == "accept" else "edited",
        )
        if produces_pair
        else None,
    }
    fields.update(kwargs)
    return ReviewDecision(**fields)  # type: ignore[arg-type]


class TestCandidate:
    def test_a_candidate_may_carry_only_marginal_labels(self):
        # Deliberately weaker than GoldenPair. A generator that proposes
        # nothing relevant has produced a bad proposal, and the review step
        # exists to catch exactly that -- refusing to store it would hide the
        # evidence the rejection rate is supposed to provide.
        assert candidate(labels=(label(gain=1),)).labels[0].gain == 1

    def test_requires_at_least_one_label(self):
        with pytest.raises(ValidationError):
            candidate(labels=())

    def test_rejects_an_inverted_source_span(self):
        with pytest.raises(ValidationError, match="source_char_end"):
            candidate(source_char_start=500, source_char_end=10)

    def test_requires_its_generator_and_prompt_version(self):
        with pytest.raises(ValidationError):
            candidate(generator_model="")
        with pytest.raises(ValidationError):
            candidate(prompt_version="")


class TestReviewDecision:
    def test_an_acceptance_carries_a_pair(self):
        assert decision().pair is not None

    def test_a_rejection_must_not_carry_a_pair(self):
        with pytest.raises(ValidationError, match="must not carry a pair"):
            ReviewDecision(
                candidate_id="c-001",
                action="reject",
                reviewer="rahul",
                decided_at="2026-09-13T10:00:00+00:00",
                reason="echoes the passage verbatim",
                pair=reviewed_pair(),
            )

    def test_a_rejection_requires_a_reason(self):
        with pytest.raises(ValidationError, match="reason"):
            decision(action="reject")

    def test_a_rejection_with_a_reason_is_valid(self):
        assert decision(action="reject", reason="echoes the passage verbatim").reason

    def test_a_skip_needs_no_reason(self):
        # A skip is "come back to this", not a judgment.
        assert decision(action="skip").action == "skip"

    def test_an_acceptance_must_not_claim_an_edit_in_its_pair(self):
        with pytest.raises(ValidationError, match="review_action"):
            ReviewDecision(
                candidate_id="c-001",
                action="accept",
                reviewer="rahul",
                decided_at="2026-09-13T10:00:00+00:00",
                pair=reviewed_pair(action="edited"),
            )

    def test_the_reviewer_must_match_the_pair(self):
        with pytest.raises(ValidationError, match="reviewer"):
            ReviewDecision(
                candidate_id="c-001",
                action="accept",
                reviewer="someone-else",
                decided_at="2026-09-13T10:00:00+00:00",
                pair=reviewed_pair(),
            )

    def test_the_candidate_id_must_match_the_pair(self):
        # Otherwise a pair can be attributed to a candidate it did not come
        # from, and the provenance trail silently forks.
        with pytest.raises(ValidationError, match="candidate_id"):
            ReviewDecision(
                candidate_id="c-999",
                action="accept",
                reviewer="rahul",
                decided_at="2026-09-13T10:00:00+00:00",
                pair=reviewed_pair("c-001"),
            )

    def test_a_hand_authored_pair_needs_no_candidate(self):
        hand = ReviewDecision(
            candidate_id=None,
            action="accept",
            reviewer="rahul",
            decided_at="2026-09-13T10:00:00+00:00",
            pair=reviewed_pair(None),
        )
        assert hand.pair is not None
        assert hand.pair.provenance.origin == "human"
        assert hand.subject == "query:q-001"

    def test_a_rejection_must_name_a_candidate(self):
        with pytest.raises(ValidationError, match="candidate_id"):
            decision(None, "reject", reason="no")


class TestMaterialize:
    def test_collects_accepted_and_edited_pairs(self):
        golden = materialize(
            [
                decision("c-1", "accept", query_id="q-1"),
                decision("c-2", "edit", query_id="q-2"),
                decision("c-3", "reject", reason="verbatim echo"),
                decision("c-4", "skip"),
            ]
        )
        assert [p.query_id for p in golden.pairs] == ["q-1", "q-2"]

    def test_the_last_decision_on_a_candidate_wins(self):
        golden = materialize(
            [decision("c-1", "accept", query_id="q-1"), decision("c-1", "reject", reason="wrong")]
        )
        assert golden.pairs == ()

    def test_a_reversal_can_reinstate_a_pair(self):
        golden = materialize(
            [
                decision("c-1", "reject", reason="too vague"),
                decision("c-1", "edit", query_id="q-1"),
            ]
        )
        assert [p.query_id for p in golden.pairs] == ["q-1"]

    def test_an_empty_log_materializes_an_empty_set(self):
        assert materialize([]).pairs == ()


class TestStore:
    def test_a_decision_survives_a_round_trip(self, tmp_path: Path):
        append_decision(decision(), tmp_path)
        assert read_review_log(tmp_path) == (decision(),)

    def test_decisions_append_rather_than_replace(self, tmp_path: Path):
        append_decision(decision("c-1", "accept", query_id="q-1"), tmp_path)
        append_decision(decision("c-1", "reject", reason="wrong"), tmp_path)
        log = read_review_log(tmp_path)
        assert len(log) == 2
        assert [d.action for d in log] == ["accept", "reject"]

    def test_reading_a_missing_log_is_empty_not_an_error(self, tmp_path: Path):
        # Before the first review there is no file, and that is the normal
        # starting state rather than a failure.
        assert read_review_log(tmp_path) == ()

    def test_a_corrupt_log_line_names_the_file_and_line(self, tmp_path: Path):
        append_decision(decision(), tmp_path)
        (tmp_path / REVIEW_LOG_FILENAME).open("a", encoding="utf-8").write("{not json\n")
        with pytest.raises(GoldenStoreError, match=r"review_log\.jsonl line 2"):
            read_review_log(tmp_path)

    def test_candidates_round_trip(self, tmp_path: Path):
        append_candidates([candidate("c-1"), candidate("c-2")], tmp_path)
        assert [c.candidate_id for c in read_candidates(tmp_path)] == ["c-1", "c-2"]

    def test_golden_set_round_trips(self, tmp_path: Path):
        golden = materialize([decision("c-1", "accept", query_id="q-1")])
        write_golden_set(golden, tmp_path)
        assert read_golden_set(tmp_path).golden_set_hash == golden.golden_set_hash

    def test_writing_the_same_set_twice_is_byte_identical(self, tmp_path: Path):
        # A re-materialization with no review activity must show an empty git
        # diff, or nobody can tell a real change from noise.
        golden = materialize([decision("c-2", "accept", query_id="q-2"), decision("c-1")])
        write_golden_set(golden, tmp_path)
        first = (tmp_path / "golden_set.jsonl").read_bytes()
        write_golden_set(golden, tmp_path)
        assert (tmp_path / "golden_set.jsonl").read_bytes() == first

    def test_rebuild_replays_the_log_onto_disk(self, tmp_path: Path):
        append_decision(decision("c-1", "accept", query_id="q-1"), tmp_path)
        append_decision(decision("c-2", "reject", reason="echo"), tmp_path)
        golden = rebuild_golden_set(tmp_path)
        assert [p.query_id for p in golden.pairs] == ["q-1"]
        assert read_golden_set(tmp_path).pairs == golden.pairs

    def test_an_empty_golden_set_writes_an_empty_file(self, tmp_path: Path):
        write_golden_set(materialize([]), tmp_path)
        assert (tmp_path / "golden_set.jsonl").read_text(encoding="utf-8") == ""


class TestReviewSummary:
    def test_reports_the_counts_the_report_quotes(self, tmp_path: Path):
        append_candidates([candidate(f"c-{i}") for i in range(4)], tmp_path)
        append_decision(decision("c-0", "accept", query_id="q-0"), tmp_path)
        append_decision(decision("c-1", "edit", query_id="q-1"), tmp_path)
        append_decision(decision("c-2", "reject", reason="verbatim echo"), tmp_path)
        append_decision(decision("c-3", "skip"), tmp_path)

        summary = review_summary(tmp_path)
        assert summary["candidates_generated"] == 4
        assert (summary["accept"], summary["edit"], summary["reject"], summary["skip"]) == (
            1,
            1,
            1,
            1,
        )

    def test_counts_superseded_decisions_separately(self, tmp_path: Path):
        # A reversal must be visible as a reversal, not vanish into the
        # current state -- that is the whole reason the log is append-only.
        append_decision(decision("c-1", "accept", query_id="q-1"), tmp_path)
        append_decision(decision("c-1", "reject", reason="on reflection, ambiguous"), tmp_path)
        summary = review_summary(tmp_path)
        assert summary["decisions_recorded"] == 2
        assert summary["subjects_decided"] == 1
        assert summary["superseded"] == 1
