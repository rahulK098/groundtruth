"""Turning a model's proposal into a candidate.

Split from the API call on purpose: this is where the bugs are, and it is
testable with no key and no network. :mod:`groundtruth.golden.generate` does
nothing but talk to Anthropic.

The important work here is **resolving quotes to absolute character offsets**.
The model is shown a windowed passage and asked to quote from it; the labels
have to point into the whole normalized document (ADR-0002). A quote that does
not appear verbatim, or appears more than once, is rejected rather than
resolved to a best guess -- a label at the wrong offsets is worse than no
label, because nothing downstream can detect it.

Rejections are **returned, not silently dropped**. The rejection rate is part
of the evidence ADR-0009 asks for, and a filter that quietly discarded a third
of its input would make "generated 150, accepted 62" a lie by omission.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from groundtruth.golden.candidates import Candidate
from groundtruth.golden.models import CATEGORIES, MAX_GAIN, RelevanceLabel
from groundtruth.golden.sampling import SampledPassage

#: A span longer than this is not a span, it is the passage. Span-level Recall
#: would be satisfied by any chunk overlapping it, making the question
#: trivially easy and biasing every configuration upward at once.
MAX_QUOTE_CHARS: Final[int] = 600

#: Below this a "quote" cannot identify a holding -- it is a fragment that
#: will match in dozens of places.
MIN_QUOTE_CHARS: Final[int] = 20

#: A *proposal* may not grade a span 0, though a ``RelevanceLabel`` may. The
#: asymmetry is deliberate: "not relevant" is a real judgment when a human
#: makes it, and incoherent when the generator makes it -- it was asked to
#: quote spans that answer its own question. The tool schema says 1-3 too, so
#: a 0 here means the model ignored the schema and the proposal is suspect.
MIN_PROPOSED_GAIN: Final[int] = 1


@dataclass(frozen=True, slots=True)
class RejectedProposal:
    """A proposal that could not become a candidate, and why."""

    passage_id: str
    code: str
    detail: str


def resolve_quote(quote: str, passage: SampledPassage) -> tuple[int, int] | RejectedProposal:
    if len(quote) < MIN_QUOTE_CHARS:
        return RejectedProposal(
            passage.passage_id, "quote-too-short", f"{len(quote)} characters: {quote!r}"
        )
    if len(quote) > MAX_QUOTE_CHARS:
        return RejectedProposal(
            passage.passage_id,
            "quote-too-long",
            f"{len(quote)} characters; a span that size is the passage, not a holding",
        )

    first = passage.text.find(quote)
    if first < 0:
        return RejectedProposal(
            passage.passage_id, "quote-not-found", f"not a verbatim substring: {quote[:80]!r}"
        )
    if passage.text.find(quote, first + 1) >= 0:
        # Ambiguous: resolving to the first occurrence would be a guess, and a
        # label at the wrong offsets is undetectable downstream.
        return RejectedProposal(
            passage.passage_id, "quote-ambiguous", f"appears more than once: {quote[:80]!r}"
        )

    return (passage.char_start + first, passage.char_start + first + len(quote))


def _labels_from_spans(
    spans: Sequence[Mapping[str, Any]], passage: SampledPassage
) -> tuple[RelevanceLabel, ...] | RejectedProposal:
    labels: list[RelevanceLabel] = []
    for span in spans:
        quote = str(span.get("quote", ""))
        gain = span.get("gain")
        if not isinstance(gain, int) or not MIN_PROPOSED_GAIN <= gain <= MAX_GAIN:
            return RejectedProposal(passage.passage_id, "bad-gain", f"gain was {gain!r}")

        resolved = resolve_quote(quote, passage)
        if isinstance(resolved, RejectedProposal):
            return resolved

        char_start, char_end = resolved
        labels.append(
            RelevanceLabel(
                doc_id=passage.doc_id,
                char_start=char_start,
                char_end=char_end,
                gain=gain,
                quote=quote,
            )
        )

    if not labels:
        return RejectedProposal(passage.passage_id, "no-spans", "the proposal quoted nothing")

    keys = [label.span_key for label in labels]
    if len(set(keys)) != len(keys):
        return RejectedProposal(passage.passage_id, "duplicate-span", "two labels on one span")

    return tuple(labels)


def candidate_from_proposal(
    raw: Mapping[str, Any],
    passage: SampledPassage,
    *,
    candidate_id: str,
    generator_model: str,
    prompt_version: str,
    generated_at: str = "",
) -> Candidate | RejectedProposal:
    """Validate one model proposal and resolve its quotes to document offsets."""
    if raw.get("decline"):
        return RejectedProposal(
            passage.passage_id,
            "declined",
            str(raw.get("decline_reason", "the model declined the passage")),
        )

    query = str(raw.get("query", "")).strip()
    if not query:
        return RejectedProposal(passage.passage_id, "no-query", "the proposal had no question")

    category = raw.get("category")
    if category not in CATEGORIES:
        return RejectedProposal(passage.passage_id, "bad-category", f"category was {category!r}")

    spans = raw.get("spans")
    if not isinstance(spans, Sequence) or isinstance(spans, str) or not spans:
        return RejectedProposal(passage.passage_id, "no-spans", "the proposal quoted nothing")

    labels = _labels_from_spans(spans, passage)
    if isinstance(labels, RejectedProposal):
        return labels

    return Candidate(
        candidate_id=candidate_id,
        query=query,
        category=category,
        labels=labels,
        generator_model=generator_model,
        prompt_version=prompt_version,
        generated_at=generated_at,
        source_doc_id=passage.doc_id,
        source_char_start=passage.char_start,
        source_char_end=passage.char_end,
    )
