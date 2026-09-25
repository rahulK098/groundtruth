"""Span -> chunk resolution (ADR-0002).

> A chunk is relevant to a span iff they share a `doc_id` and overlap by at
> least `max(64, 0.5 x len(span))` characters.

This is the one seam every metric in the scorer goes through, so it is worth
pinning against worked numbers rather than trusting the metric tests to catch
a mistake here indirectly.
"""

from __future__ import annotations

import pytest

from groundtruth.golden.models import RelevanceLabel
from groundtruth.retrieval.models import PassageScores, RetrievedPassage
from groundtruth.scoring.span_resolution import chunk_covers_span, min_overlap_chars


def passage(
    doc_id: str = "cl-1", start: int = 0, end: int = 100, rank: int = 1
) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=f"{doc_id}:{start}-{end}",
        doc_id=doc_id,
        text="x" * (end - start),
        char_start=start,
        char_end=end,
        rank=rank,
        scores=PassageScores(),
    )


def span(doc_id: str = "cl-1", start: int = 0, end: int = 100, gain: int = 3) -> RelevanceLabel:
    return RelevanceLabel(
        doc_id=doc_id, char_start=start, char_end=end, gain=gain, quote="x" * (end - start)
    )


class TestMinOverlapChars:
    def test_floors_at_64_for_a_short_span(self):
        assert min_overlap_chars(20) == 64.0

    def test_is_half_the_span_once_that_exceeds_the_floor(self):
        assert min_overlap_chars(200) == 100.0

    def test_exactly_at_the_floor_boundary(self):
        assert min_overlap_chars(128) == 64.0


class TestChunkCoversSpan:
    def test_identical_spans_cover(self):
        assert chunk_covers_span(passage(start=100, end=300), span(start=100, end=300)) is True

    def test_disjoint_spans_on_the_same_document_do_not_cover(self):
        assert chunk_covers_span(passage(start=0, end=100), span(start=500, end=700)) is False

    def test_different_documents_never_cover_regardless_of_offsets(self):
        # Same numeric offsets, different documents -- offsets are meaningless
        # across documents.
        assert (
            chunk_covers_span(
                passage(doc_id="cl-1", start=0, end=300), span(doc_id="cl-2", start=0, end=300)
            )
            is False
        )

    def test_overlap_exactly_at_the_required_minimum_covers(self):
        # A 200-char span needs 100 chars of overlap. A chunk covering
        # exactly [100:300) against a span at [0:200) overlaps [100:200) = 100.
        assert chunk_covers_span(passage(start=100, end=300), span(start=0, end=200)) is True

    def test_overlap_one_char_short_of_the_minimum_does_not_cover(self):
        assert chunk_covers_span(passage(start=101, end=300), span(start=0, end=200)) is False

    def test_a_long_span_worked_example(self):
        # span [0:1000) -> min_overlap = 500.
        # chunk [500:1000) overlaps [500:1000) = 500 chars -> covers.
        assert chunk_covers_span(passage(start=500, end=1000), span(start=0, end=1000)) is True
        # chunk [501:1000) overlaps 499 chars -> does not cover.
        assert chunk_covers_span(passage(start=501, end=1000), span(start=0, end=1000)) is False

    def test_a_short_span_uses_the_64_char_floor_not_half_its_length(self):
        # span [0:40) -> half-length would be 20, but the floor is 64.
        # A chunk overlapping all 40 characters still falls short of 64.
        assert chunk_covers_span(passage(start=0, end=40), span(start=0, end=40)) is False

    def test_full_containment_still_needs_the_64_char_floor(self):
        # span [50:90) is 40 chars, below the floor -- full containment still
        # only yields 40 chars of overlap, short of the 64 required.
        assert chunk_covers_span(passage(start=0, end=200), span(start=50, end=90)) is False

    def test_a_chunk_entirely_containing_a_long_enough_span_covers_it(self):
        assert chunk_covers_span(passage(start=0, end=300), span(start=50, end=150)) is True

    @pytest.mark.parametrize(
        "chunk_start, chunk_end, span_start, span_end",
        [
            (0, 100, 100, 200),  # touching, no overlap
            (200, 300, 0, 100),  # chunk entirely after the span
        ],
    )
    def test_touching_or_disjoint_spans_do_not_cover(
        self, chunk_start: int, chunk_end: int, span_start: int, span_end: int
    ):
        assert (
            chunk_covers_span(
                passage(start=chunk_start, end=chunk_end), span(start=span_start, end=span_end)
            )
            is False
        )
