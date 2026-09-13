"""The whole retrieval path, end to end, on the five-document mini corpus.

Real documents, the real vendored tokenizer, the real chunker, the real
indexes and the real retriever. Only the encoder is a stand-in.

This is the test that would catch a stage wired to the wrong input -- the
class of bug every unit test above passes straight through, because each one
only ever sees its own stage.
"""

from __future__ import annotations

import pytest

from groundtruth.chunking.pipeline import chunk_corpus
from groundtruth.retrieval.build import UnsupportedBackendError, build_retriever
from tests.fixtures.mini_corpus import HashingEmbedder, mini_config, mini_documents

DOCUMENTS = mini_documents()
BY_ID = {doc.doc_id: doc for doc in DOCUMENTS}

# Each query names a topic only one of the five opinions covers.
TOPICS = [
    ("what is the standard for granting summary judgment", "mini-001"),
    ("minimum contacts and purposeful availment by a nonresident", "mini-002"),
    ("when is an officer entitled to qualified immunity", "mini-003"),
    ("warrantless search of a vehicle incident to arrest", "mini-004"),
    ("whether consideration supports an enforceable contract", "mini-005"),
]


def make(mode: str = "dense", *, top_k: int = 5, **kwargs):
    config = mini_config(name=f"mini_{mode}", retrieval_mode=mode, top_k=top_k, **kwargs)
    return build_retriever(config, DOCUMENTS, HashingEmbedder())


class TestIndexConstruction:
    def test_every_chunk_is_indexed(self):
        config = mini_config()
        expected = len(chunk_corpus(DOCUMENTS, config.chunking))
        retriever = build_retriever(config, DOCUMENTS, HashingEmbedder())
        assert retriever.vector_index.count == expected
        assert expected > len(DOCUMENTS)  # multi-chunk documents are exercised

    def test_the_lexical_index_covers_the_same_chunks(self):
        retriever = make("hybrid")
        assert retriever.lexical_index is not None
        assert retriever.lexical_index.count == retriever.vector_index.count

    def test_a_dense_config_builds_no_lexical_index(self):
        assert make("dense").lexical_index is None

    def test_the_postgres_lexical_backend_is_not_available_here(self):
        # pg_fts is the service path (ADR-0001). Silently substituting BM25
        # for it would make a `pg_fts` result file a lie about what ran.
        config = mini_config(retrieval_mode="hybrid")
        pg = config.model_copy(
            update={"lexical": config.lexical.model_copy(update={"backend": "pg_fts"})}
        )
        with pytest.raises(UnsupportedBackendError, match="pg_fts"):
            build_retriever(pg, DOCUMENTS, HashingEmbedder())


@pytest.mark.parametrize(("query", "expected_doc"), TOPICS)
class TestRetrievalQuality:
    def test_dense_retrieval_finds_the_right_opinion(self, query: str, expected_doc: str):
        assert make("dense").retrieve(query).passages[0].doc_id == expected_doc

    def test_hybrid_retrieval_finds_the_right_opinion(self, query: str, expected_doc: str):
        assert make("hybrid").retrieve(query).passages[0].doc_id == expected_doc


class TestInvariants:
    def test_every_passage_span_matches_its_source_document(self):
        # The invariant span-level scoring rests on (ADR-0002). If a passage's
        # text ever stops matching document.text[start:end], every relevance
        # label silently resolves against the wrong characters.
        retriever = make("hybrid")
        for query, _ in TOPICS:
            for passage in retriever.retrieve(query).passages:
                source = BY_ID[passage.doc_id].text
                assert passage.text == source[passage.char_start : passage.char_end]

    def test_results_are_never_longer_than_top_k(self):
        result = make("dense", top_k=3).retrieve("summary judgment")
        assert len(result.passages) == 3

    def test_no_chunk_appears_twice(self):
        passages = make("hybrid").retrieve("summary judgment standard").passages
        assert len({p.chunk_id for p in passages}) == len(passages)

    def test_two_independent_builds_rank_identically(self):
        # Reproducibility is the product. Two retrievers built from the same
        # inputs must not differ in a single rank or score.
        left = make("hybrid").retrieve("qualified immunity clearly established law")
        right = make("hybrid").retrieve("qualified immunity clearly established law")
        assert [(p.chunk_id, p.rank) for p in left.passages] == [
            (p.chunk_id, p.rank) for p in right.passages
        ]
        assert [p.scores.fused for p in left.passages] == [p.scores.fused for p in right.passages]

    def test_hybrid_and_dense_do_not_produce_identical_rankings(self):
        # If they did, the lexical arm would be decorative and the whole
        # hybrid-vs-dense comparison would be measuring nothing.
        dense = make("dense")
        hybrid = make("hybrid")
        differing = [
            query
            for query, _ in TOPICS
            if [p.chunk_id for p in dense.retrieve(query).passages]
            != [p.chunk_id for p in hybrid.retrieve(query).passages]
        ]
        assert differing

    def test_an_empty_corpus_is_refused(self):
        # A retriever over nothing answers every query with nothing, which
        # scores 0.0 on every metric and looks exactly like a catastrophic
        # regression rather than a missing corpus.
        with pytest.raises(UnsupportedBackendError, match="no chunks"):
            build_retriever(mini_config(), (), HashingEmbedder())
