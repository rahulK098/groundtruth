"""Review logic, kept out of the interaction loop.

Everything here is pure: which candidates still need a decision, how one is
rendered for a human, how an edited candidate is parsed back, and what a
decision turns into. The CLI in :mod:`groundtruth.cli.golden` supplies the
keystrokes and the editor, and nothing else.

That split exists because the review is the project's critical path. Two hours
of human judgment is expensive, and a bug in "which candidate comes next" or
"what did the reviewer actually change" costs some of it -- so those parts are
testable without a terminal.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import yaml

from groundtruth.golden.candidates import Candidate, ReviewDecision, query_id_for
from groundtruth.golden.models import CATEGORIES, GoldenPair, Provenance, RelevanceLabel
from groundtruth.golden.overlap import EchoSignal, echo_signal
from groundtruth.golden.proposals import RejectedProposal, resolve_quote
from groundtruth.golden.sampling import SampledPassage

#: Decisions that settle a candidate. A skip explicitly does not.
TERMINAL_ACTIONS: Final[frozenset[str]] = frozenset({"accept", "edit", "reject"})

#: Shown beside every candidate. Not thresholds -- the reviewer decides. An
#: automatic cutoff would be one more untuned knob quietly shaping the data.
ECHO_OVERLAP_NOTE: Final[float] = 0.8
ECHO_RUN_NOTE: Final[int] = 6


class ReviewError(Exception):
    """A candidate could not be edited or turned into a pair."""


def pending(
    candidates: Sequence[Candidate],
    decisions: Iterable[ReviewDecision],
    *,
    include_skipped: bool = True,
) -> tuple[Candidate, ...]:
    """Candidates still awaiting a decision, in candidate order.

    A skipped candidate comes back by default: "skip" means *come back to
    this*, and a queue that quietly dropped them would turn indecision into
    silent exclusion from the golden set.
    """
    latest: dict[str, ReviewDecision] = {}
    for decision in decisions:
        latest[decision.subject] = decision

    settled = {
        subject
        for subject, decision in latest.items()
        if decision.action in TERMINAL_ACTIONS
        or (decision.action == "skip" and not include_skipped)
    }
    return tuple(c for c in candidates if c.candidate_id not in settled)


def source_passage(candidate: Candidate, document_text: str) -> SampledPassage:
    """Rebuild the window the generator was shown."""
    return SampledPassage(
        doc_id=candidate.source_doc_id,
        char_start=candidate.source_char_start,
        char_end=candidate.source_char_end,
        text=document_text[candidate.source_char_start : candidate.source_char_end],
    )


def candidate_echo(candidate: Candidate, passage: SampledPassage) -> EchoSignal:
    return echo_signal(candidate.query, passage.text)


def echo_warning(signal: EchoSignal) -> str:
    """A note when a candidate looks lifted from its passage, else empty.

    Advisory only. ADR-0009 puts a human in this loop precisely because the
    judgment is not reducible to a cutoff.
    """
    reasons = []
    if signal.token_overlap >= ECHO_OVERLAP_NOTE:
        reasons.append(f"{signal.token_overlap:.0%} of query tokens appear in the passage")
    if signal.longest_common_run >= ECHO_RUN_NOTE:
        reasons.append(f"{signal.longest_common_run} consecutive tokens shared verbatim")
    if not reasons:
        return ""
    return "possible echo: " + "; ".join(reasons)


def _highlight(passage: SampledPassage, labels: Sequence[RelevanceLabel]) -> str:
    """Render the passage with each labeled span marked.

    Marked in place rather than listed separately so the reviewer sees the
    span *in context* -- whether a quote is the holding or a fragment of one
    is not visible from the quote alone.
    """
    marks: list[tuple[int, int, int]] = []
    for label in labels:
        start = label.char_start - passage.char_start
        end = label.char_end - passage.char_start
        if 0 <= start < end <= len(passage.text):
            marks.append((start, end, label.gain))

    out: list[str] = []
    cursor = 0
    for start, end, gain in sorted(marks):
        if start < cursor:  # pragma: no cover - overlapping spans are rejected upstream
            continue
        out.append(passage.text[cursor:start])
        out.append(f"[[gain {gain}>>{passage.text[start:end]}<<]]")
        cursor = end
    out.append(passage.text[cursor:])
    return "".join(out)


def render_candidate(
    candidate: Candidate, passage: SampledPassage, position: int, total: int
) -> str:
    """The screen a reviewer decides from."""
    signal = candidate_echo(candidate, passage)
    warning = echo_warning(signal)

    lines = [
        f"[{position}/{total}]  {candidate.candidate_id}  ({candidate.source_doc_id} "
        f"{candidate.source_char_start}-{candidate.source_char_end})",
        "",
        f"QUERY     {candidate.query}",
        f"CATEGORY  {candidate.category}",
        f"ECHO      overlap {signal.token_overlap:.0%} of {signal.query_tokens} tokens, "
        f"longest shared run {signal.longest_common_run}",
    ]
    if warning:
        lines.append(f"          {warning}")

    lines += ["", "PASSAGE (labeled spans marked)", "", _highlight(passage, candidate.labels), ""]
    for label in candidate.labels:
        lines.append(f"  gain {label.gain}  [{label.char_start}:{label.char_end}]  {label.quote}")
    return "\n".join(lines)


# --- editing ----------------------------------------------------------------

_EDIT_HEADER = """\
# Edit the query, category or spans, then save and close.
#
# Every quote must appear EXACTLY ONCE, character for character, in the
# passage shown during review -- offsets are re-derived from the quote, so a
# quote that no longer matches is rejected rather than guessed at.
#
# gain:  3 dispositive  2 relevant  1 partial  0 not relevant
# category: {categories}
"""


def to_editable(candidate: Candidate) -> str:
    """Render a candidate as the YAML a reviewer edits."""
    body = {
        "query": candidate.query,
        "category": candidate.category,
        "spans": [{"gain": label.gain, "quote": label.quote} for label in candidate.labels],
    }
    header = _EDIT_HEADER.format(categories=" | ".join(CATEGORIES))
    return header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=88)


def from_editable(text: str, candidate: Candidate, passage: SampledPassage) -> Candidate:
    """Parse an edited candidate, re-resolving every quote against the passage."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ReviewError(f"the edited candidate is not valid YAML: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ReviewError("the edited candidate must be a YAML mapping")

    query = str(raw.get("query", "")).strip()
    if not query:
        raise ReviewError("the edited candidate has no query")

    category = raw.get("category")
    if category not in CATEGORIES:
        raise ReviewError(f"unknown category {category!r}; expected one of {', '.join(CATEGORIES)}")

    spans = raw.get("spans")
    if not isinstance(spans, Sequence) or isinstance(spans, str) or not spans:
        raise ReviewError("the edited candidate has no spans")

    labels = tuple(_label_from_edit(span, passage) for span in spans)
    return candidate.model_copy(update={"query": query, "category": category, "labels": labels})


def _label_from_edit(span: Any, passage: SampledPassage) -> RelevanceLabel:
    if not isinstance(span, Mapping):
        raise ReviewError(f"each span must be a mapping with 'gain' and 'quote', got {span!r}")

    quote = str(span.get("quote", ""))
    gain = span.get("gain")
    if not isinstance(gain, int):
        raise ReviewError(f"gain must be an integer, got {gain!r}")

    resolved = resolve_quote(quote, passage)
    if isinstance(resolved, RejectedProposal):
        raise ReviewError(f"{resolved.code}: {resolved.detail}")

    char_start, char_end = resolved
    return RelevanceLabel(
        doc_id=passage.doc_id, char_start=char_start, char_end=char_end, gain=gain, quote=quote
    )


def edited_fields(original: Candidate, revised: Candidate) -> tuple[str, ...]:
    """Which fields the reviewer actually changed.

    Recorded on the pair, because the report distinguishes accepted-as-proposed
    from edited -- and "edited" with nothing listed would be indistinguishable
    from an acceptance.
    """
    changed = []
    if original.query != revised.query:
        changed.append("query")
    if original.category != revised.category:
        changed.append("category")
    if original.labels != revised.labels:
        changed.append("labels")
    return tuple(changed)


# --- decisions --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReviewContext:
    """Who is reviewing, and when. Supplied by the caller so replay is testable."""

    reviewer: str
    decided_at: str


def accept(
    candidate: Candidate,
    context: ReviewContext,
    *,
    revised: Candidate | None = None,
    seconds: float | None = None,
) -> ReviewDecision:
    """Turn a candidate into a decision carrying its golden pair.

    Passing ``revised`` records an edit; the changed fields are derived rather
    than declared, so they cannot disagree with what was actually changed.
    """
    final = revised or candidate
    changed = edited_fields(candidate, final) if revised is not None else ()
    action = "edit" if changed else "accept"

    pair = GoldenPair(
        query_id=query_id_for(candidate.candidate_id),
        query=final.query,
        category=final.category,
        labels=final.labels,
        provenance=Provenance(
            origin="llm",
            generator_model=candidate.generator_model,
            prompt_version=candidate.prompt_version,
            source_candidate_id=candidate.candidate_id,
            reviewer=context.reviewer,
            reviewed_at=context.decided_at,
            review_action="edited" if changed else "accepted",
            edited_fields=changed,
            review_seconds=seconds,
        ),
    )
    return ReviewDecision(
        candidate_id=candidate.candidate_id,
        action=action,
        reviewer=context.reviewer,
        decided_at=context.decided_at,
        review_seconds=seconds,
        pair=pair,
    )


def reject(
    candidate: Candidate, context: ReviewContext, reason: str, *, seconds: float | None = None
) -> ReviewDecision:
    if not reason.strip():
        raise ReviewError("a rejection needs a reason")
    return ReviewDecision(
        candidate_id=candidate.candidate_id,
        action="reject",
        reviewer=context.reviewer,
        decided_at=context.decided_at,
        review_seconds=seconds,
        reason=reason.strip(),
    )


def skip(
    candidate: Candidate, context: ReviewContext, *, seconds: float | None = None
) -> ReviewDecision:
    return ReviewDecision(
        candidate_id=candidate.candidate_id,
        action="skip",
        reviewer=context.reviewer,
        decided_at=context.decided_at,
        review_seconds=seconds,
    )
