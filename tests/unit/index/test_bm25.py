"""In-process Okapi BM25.

The claim this index exists to support is that the evaluation path's lexical
arm has a **real IDF term**, which Postgres ``ts_rank_cd`` does not
(ADR-0007). So the tests below are not "it returns something plausible": one
of them is arithmetic written out by hand, and another fails if IDF is
removed.
"""

from __future__ import annotations

import math

import pytest

from groundtruth.index.bm25 import Bm25Index, Bm25IndexError

# Three documents, used for the hand-computed scores below.
#   d1  the court granted summary judgment        5 tokens
#   d2  the court denied the motion               5 tokens
#   d3  summary judgment standard                 3 tokens
TINY = {
    "d1": "the court granted summary judgment",
    "d2": "the court denied the motion",
    "d3": "summary judgment standard",
}


def build(docs: dict[str, str], **kwargs: float) -> Bm25Index:
    return Bm25Index(tuple(docs), tuple(docs.values()), **kwargs)  # type: ignore[arg-type]


class TestKnownAnswer:
    """Worked by hand so a reviewer can check the implementation, not trust it.

    N = 3, avgdl = (5 + 5 + 3) / 3 = 13/3.
    Query "summary judgment"; both terms have df = 2, so both share

        idf = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6) = 0.4700036

    With k1 = 1.2, b = 0.75 and tf = 1 for every match:

        d1 (dl=5):  norm = 1.2 * (0.25 + 0.75 * 5 / (13/3)) = 1.3384615
                    per term = idf * 2.2 / (1 + 1.3384615) = 0.4421745
                    total    = 0.8843489
        d3 (dl=3):  norm = 1.2 * (0.25 + 0.75 * 3 / (13/3)) = 0.9230769
                    per term = idf * 2.2 / (1 + 0.9230769) = 0.5376842
                    total    = 1.0753683

    d3 outranks d1 purely on length normalization: the same evidence in a
    shorter document is stronger evidence.
    """

    def test_scores_match_the_hand_computation(self):
        results = build(TINY).search("summary judgment", 10)

        assert [r.chunk_id for r in results] == ["d3", "d1"]
        assert results[0].score == pytest.approx(1.0753683, rel=1e-6)
        assert results[1].score == pytest.approx(0.8843489, rel=1e-6)

    def test_documents_matching_no_query_term_are_not_candidates(self):
        # d2 shares no term with the query. Returning it with score 0 would
        # hand fusion an arbitrary rank for a document with no evidence at all.
        assert "d2" not in {r.chunk_id for r in build(TINY).search("summary judgment", 10)}


class TestIdf:
    def test_a_rare_term_outweighs_a_repeated_common_one(self):
        # This test fails if the IDF term is dropped. Ten documents contain
        # "court"; exactly one contains "estoppel". A document repeating the
        # ubiquitous term four times must still lose to the one carrying the
        # discriminating term.
        docs = {f"filler-{i}": "court order filed" for i in range(8)}
        docs["repeats-common"] = "court court court court"
        docs["has-rare"] = "court estoppel"

        results = build(docs).search("court estoppel", 10)

        assert results[0].chunk_id == "has-rare"
        scores = {r.chunk_id: r.score for r in results}
        assert scores["has-rare"] > 10 * scores["repeats-common"]

    def test_idf_is_never_negative(self):
        # The classic Robertson IDF goes negative for a term in more than half
        # the corpus, which lets a match *subtract* from a score. The
        # log(1 + x) form used here cannot.
        docs = {f"d{i}": "ubiquitous term" for i in range(10)}
        for result in build(docs).search("ubiquitous", 10):
            assert result.score > 0


class TestParameters:
    def test_b_zero_disables_length_normalization(self):
        docs = {"short": "judgment", "long": "judgment " + "filler " * 50}
        results = build(docs, b=0.0).search("judgment", 10)
        assert results[0].score == pytest.approx(results[1].score)

    def test_default_b_penalizes_the_longer_document(self):
        docs = {"short": "judgment", "long": "judgment " + "filler " * 50}
        scores = {r.chunk_id: r.score for r in build(docs).search("judgment", 10)}
        assert scores["short"] > scores["long"]

    def test_k1_saturates_term_frequency(self):
        # The point of k1: the tenth occurrence of a term must add less than
        # the second. A raw tf model would make the ratio exactly 10.
        docs = {"once": "judgment", "tenfold": "judgment " * 10}
        scores = {r.chunk_id: r.score for r in build(docs, b=0.0).search("judgment", 10)}
        assert 1.0 < scores["tenfold"] / scores["once"] < 10.0


class TestSearchBehaviour:
    def test_returns_nothing_for_a_query_with_no_known_terms(self):
        assert build(TINY).search("certiorari", 10) == ()

    def test_returns_nothing_for_an_empty_query(self):
        assert build(TINY).search("   ", 10) == ()

    def test_truncates_to_top_n(self):
        assert len(build(TINY).search("summary judgment", 1)) == 1

    def test_ties_break_by_chunk_id_ascending(self):
        docs = {"zulu": "judgment", "alpha": "judgment", "mike": "judgment"}
        results = build(docs).search("judgment", 3)
        assert [r.chunk_id for r in results] == ["alpha", "mike", "zulu"]

    def test_is_case_insensitive(self):
        plain = build(TINY).search("summary judgment", 10)
        shouted = build(TINY).search("SUMMARY JUDGMENT", 10)
        assert plain == shouted

    def test_a_repeated_query_term_is_counted_once(self):
        # Otherwise a user typing a word twice silently doubles its weight,
        # which is a property of the typist rather than of the corpus.
        once = build(TINY).search("judgment", 10)
        twice = build(TINY).search("judgment judgment", 10)
        assert once == twice

    def test_reports_its_size(self):
        assert build(TINY).count == 3


class TestConstruction:
    def test_rejects_misaligned_ids_and_texts(self):
        with pytest.raises(Bm25IndexError, match="align"):
            Bm25Index(("a", "b"), ("only one text",))

    def test_rejects_duplicate_chunk_ids(self):
        with pytest.raises(Bm25IndexError, match="duplicate"):
            Bm25Index(("a", "a"), ("one", "two"))

    def test_rejects_an_empty_corpus(self):
        # avgdl would be a division by zero, and an index that silently
        # matches nothing is worse than one that refuses to be built.
        with pytest.raises(Bm25IndexError, match="empty"):
            Bm25Index((), ())

    @pytest.mark.parametrize("k1", [0.0, -1.0, math.nan])
    def test_rejects_an_invalid_k1(self, k1: float):
        with pytest.raises(Bm25IndexError, match="k1"):
            Bm25Index(("a",), ("text",), k1=k1)

    @pytest.mark.parametrize("b", [-0.1, 1.1, math.nan])
    def test_rejects_an_invalid_b(self, b: float):
        with pytest.raises(Bm25IndexError, match="b "):
            Bm25Index(("a",), ("text",), b=b)

    def test_tolerates_a_document_with_no_tokens(self):
        # A chunk of pure punctuation is unmatchable, not invalid.
        index = Bm25Index(("a", "b"), ("---", "judgment"))
        assert [r.chunk_id for r in index.search("judgment", 10)] == ["b"]

    def test_exposes_its_chunk_ids_in_ascending_order(self):
        # The retriever uses this to confirm at startup that every candidate
        # an index can return resolves to a chunk.
        assert build(TINY).chunk_ids == ("d1", "d2", "d3")
