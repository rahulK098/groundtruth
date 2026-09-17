"""Candidates and review decisions.

The asymmetry between :class:`Candidate` and
:class:`~groundtruth.golden.models.GoldenPair` is the design, not an oversight.

A ``GoldenPair`` must carry at least one span at ``gain >= 2``, because a query
with nothing relevant is unscoreable. A ``Candidate`` need not: a generator
that proposes only marginal spans has produced a *bad proposal*, and the whole
point of the review step is that a human sees it and rejects it (ADR-0009). A
validator that silently refused to store bad candidates would hide exactly the
evidence the rejection rate is supposed to provide.

The review log is **append-only and authoritative**. ``golden_set.jsonl`` is
materialized from it, so a corrupted golden set is always recoverable, a
re-review is a new line rather than an edit, and cherry-picking shows up in
the diff.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Self

from pydantic import Field, model_validator

from groundtruth.frozen import FrozenModel
from groundtruth.golden.models import GoldenPair, GoldenSet, QueryCategory, RelevanceLabel

ReviewAction = Literal["accept", "edit", "reject", "skip"]

#: Actions that produce a pair for the golden set.
ACCEPTING_ACTIONS: frozenset[str] = frozenset({"accept", "edit"})


def query_id_for(candidate_id: str) -> str:
    """The query id a candidate becomes when accepted.

    Derived rather than allocated, so replaying the log twice produces the
    same ids. A counter would renumber the set whenever a decision earlier in
    the log was reversed, silently changing the golden-set hash and tripping
    the gate for no real reason.
    """
    stem = candidate_id[2:] if candidate_id.startswith("c-") else candidate_id
    return f"q-{stem}"


class Candidate(FrozenModel):
    """An unreviewed proposal. Never a label."""

    candidate_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    category: QueryCategory
    labels: tuple[RelevanceLabel, ...] = Field(min_length=1)

    generator_model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    generated_at: str = ""

    #: The passage the generator was shown, so the reviewer can see the
    #: proposal beside its source rather than judging it in isolation.
    source_doc_id: str = Field(min_length=1)
    source_char_start: int = Field(ge=0)
    source_char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _source_span_must_be_well_formed(self) -> Self:
        if self.source_char_end <= self.source_char_start:
            raise ValueError(
                f"source_char_end ({self.source_char_end}) must be greater than "
                f"source_char_start ({self.source_char_start})"
            )
        return self

    @model_validator(mode="after")
    def _labels_must_be_unique(self) -> Self:
        keys = [lbl.span_key for lbl in self.labels]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate labeled span in a candidate")
        return self


class ReviewDecision(FrozenModel):
    """One line of the append-only review log."""

    #: ``None`` for a pair the reviewer authored directly rather than judging.
    candidate_id: str | None = None
    action: ReviewAction
    reviewer: str = Field(min_length=1)
    decided_at: str = Field(min_length=1)
    review_seconds: float | None = Field(default=None, ge=0.0)
    #: Required when rejecting. "Rejected, no reason given" is not evidence.
    reason: str = ""
    #: Present exactly when the action produces a pair.
    pair: GoldenPair | None = None

    @model_validator(mode="after")
    def _pair_presence_must_match_the_action(self) -> Self:
        produces_pair = self.action in ACCEPTING_ACTIONS
        if produces_pair and self.pair is None:
            raise ValueError(f"action {self.action!r} requires a pair")
        if not produces_pair and self.pair is not None:
            raise ValueError(f"action {self.action!r} must not carry a pair")
        if self.action == "reject" and not self.reason.strip():
            raise ValueError(
                "a rejection requires a reason; the rejection rate is only "
                "evidence if the reasons are inspectable"
            )
        if self.action in {"reject", "skip"} and self.candidate_id is None:
            raise ValueError(f"action {self.action!r} requires a candidate_id")
        return self

    @model_validator(mode="after")
    def _pair_provenance_must_agree_with_the_decision(self) -> Self:
        if self.pair is None:
            return self

        provenance = self.pair.provenance
        expected_action = "accepted" if self.action == "accept" else "edited"
        if provenance.review_action != expected_action:
            raise ValueError(
                f"decision action {self.action!r} implies provenance "
                f"review_action {expected_action!r}, got {provenance.review_action!r}"
            )
        if provenance.reviewer != self.reviewer:
            raise ValueError(
                f"decision reviewer {self.reviewer!r} does not match the pair's "
                f"recorded reviewer {provenance.reviewer!r}"
            )
        if provenance.source_candidate_id != self.candidate_id:
            raise ValueError(
                f"decision candidate_id {self.candidate_id!r} does not match the "
                f"pair's source_candidate_id {provenance.source_candidate_id!r}"
            )
        return self

    @property
    def subject(self) -> str:
        """What this decision is about, for replay.

        A candidate id when one exists; otherwise the query id of the pair the
        reviewer wrote by hand.
        """
        if self.candidate_id is not None:
            return self.candidate_id
        if self.pair is None:  # pragma: no cover - the validators forbid this
            raise ValueError("a decision needs either a candidate_id or a pair")
        return f"query:{self.pair.query_id}"


def materialize(decisions: Iterable[ReviewDecision]) -> GoldenSet:
    """Replay the review log into the golden set it describes.

    **Last decision per subject wins**, so re-reviewing is appending a line
    rather than editing one, and reversing an earlier acceptance leaves both
    the acceptance and the reversal in the record.
    """
    latest: dict[str, ReviewDecision] = {}
    for decision in decisions:
        latest[decision.subject] = decision

    return GoldenSet(
        pairs=tuple(
            decision.pair
            for decision in latest.values()
            if decision.pair is not None and decision.action in ACCEPTING_ACTIONS
        )
    )
