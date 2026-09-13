"""Lexical analysis for the in-process BM25 index.

Deliberately minimal, and the omissions are the interesting part.

**No stopword list.** BM25's IDF term already collapses the weight of a term
that appears in most documents -- that is what IDF *is*. A hand-curated list
would be a second, untuned mechanism doing the same job, and on a legal corpus
it would be actively wrong: "no", "against" and "such" are not noise in
"summary judgment shall be granted if no genuine dispute".

**No stemming.** This is a real limitation rather than a considered win, and
it is recorded here rather than hidden: Postgres' ``english`` text search
configuration *does* stem, so when the ``pg_fts`` backend lands (ADR-0007) the
two lexical arms will differ in both stemming and IDF. That confounds the
comparison the fifth config is meant to make, and closing it is Phase 10 work.
"""

from __future__ import annotations

import re
from typing import Final

#: ``[^\W_]`` is "word character, but not underscore": unicode letters and
#: digits. A plain ``[a-z0-9]+`` would shred accented party names into
#: fragments no query could match, and ``\w`` would glue ``doc_id`` into one
#: token that matches neither ``doc`` nor ``id``.
_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Split text into lowercased lexical tokens.

    Pure and deterministic: the same string always yields the same tokens, in
    the same order, in any process. Indexing and querying both go through
    here, so a change to this function changes both sides at once -- which is
    the only safe way to change an analyzer.
    """
    return tuple(_TOKEN_PATTERN.findall(text.lower()))
