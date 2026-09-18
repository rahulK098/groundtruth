"""Hand-authored pairs.

The ~30 queries written as an attorney would phrase them never pass through
the generator (ADR-0009), so they need their own way into the review log.
They arrive as a small YAML document the reviewer fills in, and leave as a
:class:`~groundtruth.golden.candidates.ReviewDecision` with no candidate and a
pair whose provenance says ``origin="human"``.

Quotes are resolved against the **whole** normalized document rather than a
sampled window, because an author looking at the opinion has no window. The
same rules apply as for a generated proposal: a quote must appear exactly
once, or it is refused rather than guessed at (see
:mod:`groundtruth.golden.proposals`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

import yaml

from groundtruth.corpus.models import Document
from groundtruth.golden.candidates import ReviewDecision
from groundtruth.golden.models import CATEGORIES, GoldenPair, Provenance, RelevanceLabel
from groundtruth.golden.proposals import RejectedProposal, resolve_quote
from groundtruth.golden.review import ReviewContext
from groundtruth.golden.sampling import SampledPassage

#: Hand-authored ids live in their own namespace so they can never collide
#: with the ``q-NNNN`` ids derived from candidate ids.
HUMAN_QUERY_ID_PREFIX: Final[str] = "q-h-"

_HUMAN_QUERY_ID = re.compile(rf"^{re.escape(HUMAN_QUERY_ID_PREFIX)}(\d+)$")

_TEMPLATE_HEADER = """\
# Write the query as an attorney would ask it -- in your own words, NOT in
# the words of the passage. The generator is told the same thing; the point
# of hand-written queries is that they are the control.
#
# Every quote must appear EXACTLY ONCE, character for character, in the
# normalized text of the document it names. Offsets are derived from the
# quote; a quote that cannot be located is refused rather than guessed at.
#
# gain:  3 dispositive  2 relevant  1 partial  0 not relevant
#        (at least one span must be 2 or higher, or the query is unscoreable)
# category: {categories}
"""


class AuthoringError(Exception):
    """A hand-authored pair could not be parsed or resolved."""


def to_template(doc_id: str = "") -> str:
    """The YAML a reviewer fills in for one hand-written pair."""
    body = {
        "query": "",
        "category": CATEGORIES[0],
        "spans": [{"doc_id": doc_id, "gain": 3, "quote": ""}],
    }
    header = _TEMPLATE_HEADER.format(categories=" | ".join(CATEGORIES))
    return header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=88)


def _whole_document(document: Document) -> SampledPassage:
    return SampledPassage(
        doc_id=document.doc_id, char_start=0, char_end=len(document.text), text=document.text
    )


def _label_from_span(span: object, documents: Mapping[str, Document]) -> RelevanceLabel:
    if not isinstance(span, Mapping):
        raise AuthoringError(
            f"each span must be a mapping with 'doc_id', 'gain' and 'quote', got {span!r}"
        )

    doc_id = str(span.get("doc_id", "")).strip()
    document = documents.get(doc_id)
    if document is None:
        raise AuthoringError(f"span names {doc_id!r}, which is not in the corpus snapshot")

    gain = span.get("gain")
    if not isinstance(gain, int):
        raise AuthoringError(f"gain must be an integer, got {gain!r}")

    quote = str(span.get("quote", ""))
    resolved = resolve_quote(quote, _whole_document(document))
    if isinstance(resolved, RejectedProposal):
        raise AuthoringError(f"{resolved.code} in {doc_id}: {resolved.detail}")

    char_start, char_end = resolved
    try:
        return RelevanceLabel(
            doc_id=doc_id, char_start=char_start, char_end=char_end, gain=gain, quote=quote
        )
    except ValueError as exc:
        raise AuthoringError(str(exc)) from exc


def from_authored(
    text: str,
    documents: Mapping[str, Document],
    *,
    query_id: str,
    context: ReviewContext,
) -> ReviewDecision:
    """Parse a filled-in template into a decision carrying a human-origin pair."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise AuthoringError(f"the authored pair is not valid YAML: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise AuthoringError("the authored pair must be a YAML mapping")

    query = str(raw.get("query") or "").strip()
    if not query:
        raise AuthoringError("the authored pair has no query")

    category = raw.get("category")
    if category not in CATEGORIES:
        raise AuthoringError(
            f"unknown category {category!r}; expected one of {', '.join(CATEGORIES)}"
        )

    spans = raw.get("spans")
    if not isinstance(spans, Sequence) or isinstance(spans, str) or not spans:
        raise AuthoringError("the authored pair has no spans")

    labels = tuple(_label_from_span(span, documents) for span in spans)

    try:
        pair = GoldenPair(
            query_id=query_id,
            query=query,
            category=category,
            labels=labels,
            provenance=Provenance(
                origin="human",
                reviewer=context.reviewer,
                reviewed_at=context.decided_at,
                review_action="accepted",
            ),
        )
    except ValueError as exc:
        raise AuthoringError(str(exc)) from exc

    return ReviewDecision(
        candidate_id=None,
        action="accept",
        reviewer=context.reviewer,
        decided_at=context.decided_at,
        pair=pair,
    )


def next_human_query_id(decisions: Iterable[ReviewDecision]) -> str:
    """The next unused hand-authored id.

    One past the highest already in the log rather than one past the count,
    so a gap is never refilled: reusing an id would make replay collapse two
    distinct pairs into one subject.
    """
    highest = 0
    for decision in decisions:
        if decision.pair is None:
            continue
        match = _HUMAN_QUERY_ID.match(decision.pair.query_id)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{HUMAN_QUERY_ID_PREFIX}{highest + 1:03d}"
