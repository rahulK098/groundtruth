"""Rendering for ``gt search``.

The command itself needs the `models` extra to embed an arbitrary query, so it
is not exercised end to end here. The formatting is, because that is the part
with logic in it -- and because a per-stage score printed under the wrong
label would quietly mislead exactly the error analysis it exists to support.
"""

from __future__ import annotations

from groundtruth.cli.search import render
from groundtruth.retrieval.models import (
    PassageScores,
    RetrievalResult,
    RetrievedPassage,
    StageLatenciesMs,
)
from tests.fixtures.mini_corpus import mini_config

CONFIG = mini_config(name="hybrid_512", retrieval_mode="hybrid", top_k=5)


def result(*passages: RetrievedPassage) -> RetrievalResult:
    return RetrievalResult(
        query="summary judgment standard",
        config_name=CONFIG.name,
        config_hash=CONFIG.config_hash,
        passages=passages,
        latency_ms=StageLatenciesMs(embed=12.0, dense=1.0, lexical=9.0, fuse=0.4, total=23.0),
    )


def passage(
    rank: int = 1, text: str = "Summary judgment is appropriate.", **scores
) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=f"chunk-{rank}",
        doc_id="cl-112426",
        text=text,
        char_start=100,
        char_end=100 + len(text),
        rank=rank,
        scores=PassageScores(**scores),
    )


class TestRender:
    def test_reports_the_query_and_the_config_hash(self):
        out = render(result(passage(dense=0.7)), CONFIG)
        assert "summary judgment standard" in out
        # The hash, not just the name: a config edited since the run makes the
        # name alone meaningless (ADR-0010).
        assert CONFIG.config_hash in out

    def test_reports_every_stage_latency(self):
        out = render(result(passage(dense=0.7)), CONFIG)
        for stage in ("embed", "dense", "lexical", "fuse", "total"):
            assert stage in out

    def test_labels_each_arm_score_separately(self):
        out = render(result(passage(dense=0.7029, lexical=10.0365, fused=0.0303)), CONFIG)
        assert "dense 0.7029" in out
        assert "lexical 10.0365" in out
        assert "fused 0.0303" in out

    def test_omits_a_stage_that_produced_no_score(self):
        # None means "this stage did not score this passage", and printing it
        # as 0.0000 would be indistinguishable from a stage that scored zero.
        out = render(result(passage(dense=0.7)), CONFIG)
        assert "lexical" in out.splitlines()[2]  # the latency line
        assert "lexical 0" not in out
        assert "rerank" not in out.split("\n\n", maxsplit=1)[1]

    def test_shows_the_character_span(self):
        out = render(result(passage(dense=0.7)), CONFIG)
        assert "[100:132]" in out

    def test_truncates_a_long_passage_and_says_so(self):
        out = render(result(passage(dense=0.7, text="word " * 200)), CONFIG)
        assert "..." in out
        assert len(max(out.splitlines(), key=len)) < 300

    def test_collapses_newlines_inside_a_snippet(self):
        # Corpus text is full of hard-wrapped lines; leaving them in turns one
        # result into a wall and makes a top-10 unscannable.
        out = render(result(passage(dense=0.7, text="first line\nsecond line")), CONFIG)
        assert "first line second line" in out

    def test_says_so_when_nothing_was_retrieved(self):
        assert "no passages retrieved" in render(result(), CONFIG)
