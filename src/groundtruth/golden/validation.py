"""Golden-set integrity checks.

Issues are **returned, not raised**, and split into errors and warnings. That
shape is deliberate: a set under construction is legitimately incomplete, and
a checker that threw on the first missing category would be useless until the
last pair was reviewed -- which is precisely when nobody is still running it.

- **Errors** mean a label is wrong: it points outside a document, at text that
  is no longer there, or at a document that is not in the corpus. Every one of
  these silently corrupts a metric.
- **Warnings** mean the set is not finished yet. They are expected early and
  must be zero before a baseline is blessed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from groundtruth.corpus.models import Document
from groundtruth.golden.models import CATEGORIES, GoldenSet, RelevanceLabel

#: The target from the brief. A warning, never an error.
TARGET_SET_SIZE: Final[int] = 100

#: Below this, a category's numbers are noise -- which is why the gate never
#: reads per-category metrics at all (see methodology).
MIN_PER_CATEGORY: Final[int] = 12

Severity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    query_id: str | None = None

    def __str__(self) -> str:
        where = f" [{self.query_id}]" if self.query_id else ""
        return f"{self.severity.upper():7s} {self.code}{where}: {self.message}"


def _check_label_against_document(
    query_id: str, label: RelevanceLabel, documents: dict[str, Document]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    document = documents.get(label.doc_id)
    if document is None:
        issues.append(
            ValidationIssue(
                "error",
                "unknown-document",
                f"label points at {label.doc_id!r}, which is not in the corpus snapshot",
                query_id,
            )
        )
        return issues

    if label.char_end > len(document.text):
        issues.append(
            ValidationIssue(
                "error",
                "span-out-of-range",
                f"span [{label.char_start}:{label.char_end}] exceeds {label.doc_id} "
                f"({len(document.text)} characters)",
                query_id,
            )
        )
        return issues

    actual = document.text[label.char_start : label.char_end]
    if actual != label.quote:
        # The check worth the redundancy. A mismatch means the corpus was
        # re-normalized under the labels, so every offset in this document is
        # suspect -- and it would otherwise surface as an unexplained metric
        # shift weeks later.
        issues.append(
            ValidationIssue(
                "error",
                "quote-mismatch",
                f"{label.doc_id}[{label.char_start}:{label.char_end}] is "
                f"{actual[:60]!r} but the label quotes {label.quote[:60]!r}; "
                f"the corpus may have been re-normalized under this label",
                query_id,
            )
        )
    return issues


def validate_against_corpus(
    golden: GoldenSet, documents: Iterable[Document]
) -> tuple[ValidationIssue, ...]:
    """Check every label resolves to the exact text it claims."""
    by_id = {document.doc_id: document for document in documents}
    issues: list[ValidationIssue] = []
    for pair in golden.pairs:
        for label in pair.labels:
            issues.extend(_check_label_against_document(pair.query_id, label, by_id))
    return tuple(issues)


def validate_policy(
    golden: GoldenSet,
    *,
    target_size: int = TARGET_SET_SIZE,
    min_per_category: int = MIN_PER_CATEGORY,
) -> tuple[ValidationIssue, ...]:
    """Check the set is complete enough to support the claims made about it."""
    issues: list[ValidationIssue] = []

    if len(golden.pairs) < target_size:
        issues.append(
            ValidationIssue(
                "warning",
                "set-incomplete",
                f"{len(golden.pairs)} of {target_size} pairs reviewed",
            )
        )

    counts = golden.counts_by_category()
    for category in CATEGORIES:
        if counts[category] < min_per_category:
            issues.append(
                ValidationIssue(
                    "warning",
                    "category-under-populated",
                    f"{category}: {counts[category]} of {min_per_category} minimum",
                )
            )

    origins = golden.counts_by_origin()
    if golden.pairs and origins["human"] == 0:
        # Without hand-authored queries there is nothing to compare LLM-origin
        # results against, and the methodology's own check on generator bias
        # cannot be run at all.
        issues.append(
            ValidationIssue(
                "warning",
                "no-human-authored-queries",
                "every pair is LLM-proposed, so the human-vs-LLM origin "
                "breakdown -- the project's own check on generator bias -- "
                "cannot be computed",
            )
        )

    return tuple(issues)


def validate(
    golden: GoldenSet, documents: Sequence[Document], **policy: int
) -> tuple[ValidationIssue, ...]:
    """Every check, corpus first."""
    return validate_against_corpus(golden, documents) + validate_policy(golden, **policy)


def errors(issues: Iterable[ValidationIssue]) -> tuple[ValidationIssue, ...]:
    return tuple(issue for issue in issues if issue.severity == "error")
