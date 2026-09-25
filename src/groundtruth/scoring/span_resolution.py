"""Span -> chunk resolution (ADR-0002).

Relevance labels are character spans into the normalized document, never
chunk ids, so a labeled span can only be scored against a config's actual
output by resolving it to whichever chunks that config happened to produce.

> A chunk is relevant to a span iff they share a `doc_id` and overlap by at
> least `max(64, 0.5 x len(span))` characters.

This is the one seam every metric in the scorer goes through. Getting it
wrong would silently corrupt every number downstream, which is why it is its
own module with its own worked-arithmetic tests rather than inlined into the
metric functions.
"""

from __future__ import annotations

from typing import Final

from groundtruth.golden.models import RelevanceLabel
from groundtruth.retrieval.models import RetrievedPassage

#: Below this, "half the span" would demand an overlap too small to identify
#: the span rather than a fragment adjacent to it.
MIN_OVERLAP_FLOOR: Final[float] = 64.0

#: A span longer than this needs proportionally more overlap, so a chunk that
#: merely grazes one end of a long span is not credited with covering it.
MIN_OVERLAP_FRACTION: Final[float] = 0.5


def min_overlap_chars(span_char_len: int) -> float:
    """The overlap, in characters, a chunk needs to count as covering a span."""
    return max(MIN_OVERLAP_FLOOR, MIN_OVERLAP_FRACTION * span_char_len)


def chunk_covers_span(chunk: RetrievedPassage, span: RelevanceLabel) -> bool:
    """Whether ``chunk`` satisfies the overlap policy against ``span``."""
    if chunk.doc_id != span.doc_id:
        return False

    overlap = min(chunk.char_end, span.char_end) - max(chunk.char_start, span.char_start)
    if overlap <= 0:
        return False

    return overlap >= min_overlap_chars(span.char_end - span.char_start)
