"""Text normalization.

The highest-stakes unglamorous module in the project: golden-set relevance
labels are character offsets into normalized text, so changing any rule here
moves every label at once.

That is what ``NORMALIZER_VERSION`` is for. It is part of the corpus manifest
hash, which the regression gate treats as a hard failure when it changes -- so
a normalization change cannot quietly invalidate a baseline.

The guiding bias is **conservative**. Leaving noise in costs a little retrieval
quality; stripping too aggressively silently deletes content, and a label
pointing at deleted text fails in a way nobody can diagnose.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

#: Bump whenever any rule below changes. Part of the corpus manifest hash.
NORMALIZER_VERSION: Final[str] = "1"

#: Reporter star-pagination, e.g. "[*1145]". Layout, not language: left in, it
#: becomes a token that dilutes the embedding and skews BM25 term statistics.
_BRACKETED_PAGINATION: Final[re.Pattern[str]] = re.compile(r"\[\*\d+\]")

#: A whole line consisting only of "*1145". Requires digits, so a footnote
#: marker or emphasis such as "*supra" is left alone.
_LINE_PAGINATION: Final[re.Pattern[str]] = re.compile(r"(?m)^[^\S\n]*\*\d+[^\S\n]*$")

#: Any whitespace except a newline: spaces, tabs, form feeds, and the more
#: exotic Unicode separators. Newlines are handled separately because
#: paragraph structure is meaningful in an opinion.
_HORIZONTAL_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"[^\S\n]+")

_SPACE_AROUND_NEWLINE: Final[re.Pattern[str]] = re.compile(r" *\n *")

_BLANK_LINE_RUN: Final[re.Pattern[str]] = re.compile(r"\n{3,}")


def normalize_text(raw: str) -> str:
    """Normalize document text to the one form labels are offset into.

    Order matters. Artifacts are removed *before* whitespace is collapsed, so
    the gap left behind by a removed "[*1145]" closes up rather than becoming a
    double space.

    The function is idempotent -- ``normalize_text(normalize_text(x))`` equals
    ``normalize_text(x)`` -- which is what allows a snapshot to be re-read and
    re-normalized without drifting a single character offset.
    """
    # NFKC folds ligatures ("aﬃrmed" -> "affirmed"), full-width forms, and
    # non-breaking spaces. Without it, a query for "affirmed" misses a passage
    # that renders the ligature as one codepoint.
    text = unicodedata.normalize("NFKC", raw)

    # Normalize line endings before any line-anchored regex runs.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = _BRACKETED_PAGINATION.sub("", text)
    text = _LINE_PAGINATION.sub("", text)

    text = _HORIZONTAL_WHITESPACE.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE.sub("\n", text)

    # Cap consecutive newlines at one paragraph break. Collapsing them further
    # would merge a holding into the preceding recitation of facts.
    text = _BLANK_LINE_RUN.sub("\n\n", text)

    return text.strip()
