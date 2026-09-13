"""The retriever: the one shared code path above the storage primitives.

Assembled here from synthetic parts so the edge cases are reachable. The
same class is exercised against the real tokenizer and a real chunker in
``test_end_to_end.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from groundtruth.chunking.models import Chunk
from groundtruth.index.bm25 import Bm25Index
from groundtruth.index.numpy_vector import NumpyVectorIndex
from groundtruth.retrieval.pipeline import Retriever, RetrieverError
from tests.fixtures.mini_corpus import mini_config

TEXTS = {
    "c1": "summary judgment is granted",
    "c2": "personal jurisdiction requires minimum contacts",
    "c3": "qualified immunity shields the officer",
}

# Deliberately hand-built so the expected dense ranking is obvious:
# the query [1, 0, 0] matches c1 exactly, then c3, then c2.
VECTORS = {
    "c1": [1.0, 0.0, 0.0],
    "c2": [0.0, 1.0, 0.0],
    "c3": [0.6, 0.8, 0.0],
}
QUERY_VECTOR = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)


class StubEmbedder:
    """Returns one fixed vector and records exactly what it was asked to embed."""

    model_id = "test-org/stub"
    revision = "0" * 40
    normalize = True

    def __init__(self, vector: np.ndarray = QUERY_VECTOR, dimension: int = 3) -> None:
        self.vector = vector
        self.dimension = dimension
        self.seen: list[str] = []

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.seen.extend(texts)
        return np.repeat(self.vector, len(texts), axis=0)


def make_chunks() -> dict[str, Chunk]:
    chunks = {}
    for i, (chunk_id, text) in enumerate(TEXTS.items()):
        chunks[chunk_id] = Chunk(
            chunk_id=chunk_id,
            doc_id=f"doc-{i}",
            text=text,
            char_start=0,
            char_end=len(text),
            token_start=0,
            token_end=len(text.split()),
        )
    return chunks


def make_retriever(*, mode: str = "dense", top_k: int = 3, **config_kwargs) -> Retriever:
    config = mini_config(
        retrieval_mode=mode, top_k=top_k, dense_top_n=3, dimension=3, **config_kwargs
    )
    vector_index = NumpyVectorIndex(
        tuple(VECTORS), np.array(list(VECTORS.values()), dtype=np.float32)
    )
    lexical_index = Bm25Index(tuple(TEXTS), tuple(TEXTS.values())) if mode == "hybrid" else None
    return Retriever(
        config,
        embedder=StubEmbedder(),
        vector_index=vector_index,
        lexical_index=lexical_index,
        chunks=make_chunks(),
    )


class TestDenseRetrieval:
    def test_returns_passages_in_dense_score_order(self):
        result = make_retriever().retrieve("summary judgment")
        assert [p.chunk_id for p in result.passages] == ["c1", "c3", "c2"]

    def test_ranks_are_one_based_and_contiguous(self):
        result = make_retriever().retrieve("summary judgment")
        assert [p.rank for p in result.passages] == [1, 2, 3]

    def test_truncates_to_top_k(self):
        result = make_retriever(top_k=2).retrieve("summary judgment")
        assert len(result.passages) == 2

    def test_carries_the_chunk_identity_and_span(self):
        passage = make_retriever().retrieve("summary judgment").passages[0]
        assert passage.doc_id == "doc-0"
        assert passage.text == TEXTS["c1"]
        assert (passage.char_start, passage.char_end) == (0, len(TEXTS["c1"]))

    def test_reports_the_dense_score_and_no_other(self):
        # In dense mode there is nothing to fuse and nothing lexical, and
        # reporting a `fused` number there would be a fiction.
        scores = make_retriever().retrieve("summary judgment").passages[0].scores
        assert scores.dense == pytest.approx(1.0)
        assert scores.lexical is None
        assert scores.fused is None
        assert scores.rerank is None

    def test_records_the_configuration_that_produced_the_result(self):
        config = mini_config(retrieval_mode="dense", top_k=3, dense_top_n=3, dimension=3)
        result = make_retriever().retrieve("summary judgment")
        assert result.config_name == config.name
        assert result.config_hash == config.config_hash

    def test_echoes_the_query_unmodified(self):
        result = make_retriever().retrieve("  summary judgment  ")
        assert result.query == "  summary judgment  "


class TestQueryPrefix:
    def test_the_prefix_is_applied_to_the_embedder_input(self):
        # bge instructs queries and leaves passages bare. The prefix is part
        # of the embedding cache key, so applying it in the wrong place does
        # not fail loudly -- it just serves a vector for a different string.
        prefix = "Represent this sentence: "
        retriever = make_retriever(query_prefix=prefix)
        retriever.retrieve("summary judgment")
        assert retriever.embedder.seen == [prefix + "summary judgment"]  # type: ignore[attr-defined]

    def test_no_prefix_sends_the_bare_query(self):
        retriever = make_retriever()
        retriever.retrieve("summary judgment")
        assert retriever.embedder.seen == ["summary judgment"]  # type: ignore[attr-defined]


class TestHybridRetrieval:
    def test_reports_a_fused_score(self):
        result = make_retriever(mode="hybrid").retrieve("summary judgment")
        assert result.passages[0].scores.fused is not None

    def test_reports_each_arm_that_contributed(self):
        result = make_retriever(mode="hybrid").retrieve("summary judgment")
        scores = {p.chunk_id: p.scores for p in result.passages}
        # c1 is the only chunk carrying the query terms, so it is the only
        # one the lexical arm can return.
        assert scores["c1"].lexical is not None
        assert scores["c1"].dense is not None
        assert scores["c2"].lexical is None
        assert scores["c2"].dense is not None

    def test_a_lexical_only_match_still_reaches_the_results(self):
        # The point of hybrid retrieval: a chunk the dense arm never surfaced
        # can still be retrieved. The query vector points away from c1, and
        # dense_top_n = 2 cuts it from the dense candidate pool entirely --
        # so c1 can only arrive through the lexical arm.
        config = mini_config(retrieval_mode="hybrid", top_k=2, dense_top_n=2, dimension=3)
        retriever = Retriever(
            config,
            embedder=StubEmbedder(np.array([[0.0, 1.0, 0.0]], dtype=np.float32)),
            vector_index=NumpyVectorIndex(
                tuple(VECTORS), np.array(list(VECTORS.values()), dtype=np.float32)
            ),
            lexical_index=Bm25Index(tuple(TEXTS), tuple(TEXTS.values())),
            chunks=make_chunks(),
        )
        result = retriever.retrieve("summary judgment")
        found = {p.chunk_id for p in result.passages}
        assert "c1" in found  # lexical only
        assert "c2" in found  # dense only


class TestLatency:
    def test_every_executed_stage_is_timed(self):
        result = make_retriever(mode="hybrid").retrieve("summary judgment")
        assert result.latency_ms.embed >= 0.0
        assert result.latency_ms.dense >= 0.0
        assert result.latency_ms.lexical >= 0.0
        assert result.latency_ms.total >= result.latency_ms.embed

    def test_stages_that_did_not_run_are_zero(self):
        result = make_retriever().retrieve("summary judgment")
        assert result.latency_ms.lexical == 0.0
        assert result.latency_ms.fuse == 0.0
        assert result.latency_ms.rerank == 0.0


class TestValidation:
    def test_a_hybrid_config_requires_a_lexical_index(self):
        config = mini_config(retrieval_mode="hybrid", top_k=3, dense_top_n=3, dimension=3)
        with pytest.raises(RetrieverError, match="lexical index"):
            Retriever(
                config,
                embedder=StubEmbedder(),
                vector_index=NumpyVectorIndex(
                    tuple(VECTORS), np.array(list(VECTORS.values()), dtype=np.float32)
                ),
                lexical_index=None,
                chunks=make_chunks(),
            )

    def test_a_dense_config_rejects_a_lexical_index(self):
        # Dead wiring: the pipeline would never call it, and someone would
        # eventually conclude the lexical arm does nothing.
        config = mini_config(retrieval_mode="dense", top_k=3, dense_top_n=3, dimension=3)
        with pytest.raises(RetrieverError, match="never read"):
            Retriever(
                config,
                embedder=StubEmbedder(),
                vector_index=NumpyVectorIndex(
                    tuple(VECTORS), np.array(list(VECTORS.values()), dtype=np.float32)
                ),
                lexical_index=Bm25Index(tuple(TEXTS), tuple(TEXTS.values())),
                chunks=make_chunks(),
            )

    def test_a_chunk_missing_from_the_lookup_is_a_loud_failure(self):
        config = mini_config(retrieval_mode="dense", top_k=3, dense_top_n=3, dimension=3)
        chunks = make_chunks()
        del chunks["c1"]
        with pytest.raises(RetrieverError, match="not in the chunk lookup"):
            Retriever(
                config,
                embedder=StubEmbedder(),
                vector_index=NumpyVectorIndex(
                    tuple(VECTORS), np.array(list(VECTORS.values()), dtype=np.float32)
                ),
                chunks=chunks,
            )

    def test_an_embedder_of_the_wrong_dimension_is_rejected(self):
        config = mini_config(retrieval_mode="dense", top_k=3, dense_top_n=3, dimension=3)
        with pytest.raises(RetrieverError, match="dimension"):
            Retriever(
                config,
                embedder=StubEmbedder(np.zeros((1, 5), dtype=np.float32), dimension=5),
                vector_index=NumpyVectorIndex(
                    tuple(VECTORS), np.array(list(VECTORS.values()), dtype=np.float32)
                ),
                chunks=make_chunks(),
            )

    def test_rejects_an_empty_query(self):
        with pytest.raises(RetrieverError, match="empty"):
            make_retriever().retrieve("   ")


class TestUnimplementedStages:
    def test_a_config_enabling_the_reranker_is_refused(self):
        # Phase 9 work. Until then a refusal is the only honest answer:
        # running the config anyway would write a result file labelled
        # "hybrid_512_rerank" produced without a reranker.
        config = mini_config(retrieval_mode="dense", top_k=3, dense_top_n=3, dimension=3)
        with_rerank = config.model_copy(
            update={
                "reranker": config.reranker.model_copy(
                    update={
                        "enabled": True,
                        "model_id": "BAAI/bge-reranker-base",
                        "revision": "2cfc18c9415c",
                    }
                )
            }
        )
        with pytest.raises(RetrieverError, match="reranker"):
            Retriever(
                with_rerank,
                embedder=StubEmbedder(),
                vector_index=NumpyVectorIndex(
                    tuple(VECTORS), np.array(list(VECTORS.values()), dtype=np.float32)
                ),
                chunks=make_chunks(),
            )

    def test_an_index_of_the_wrong_dimension_is_rejected(self):
        config = mini_config(retrieval_mode="dense", top_k=3, dense_top_n=3, dimension=3)
        with pytest.raises(RetrieverError, match="index dimension"):
            Retriever(
                config,
                embedder=StubEmbedder(),
                vector_index=NumpyVectorIndex(("a",), np.array([[1.0, 0.0]], dtype=np.float32)),
                chunks=make_chunks(),
            )

    def test_exposes_the_configuration_it_was_built_with(self):
        assert make_retriever().config.retrieval_mode == "dense"
