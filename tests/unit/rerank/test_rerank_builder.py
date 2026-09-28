"""Building the rerank cache through the real pipeline, then serving it cache-only.

The acceptance criterion for Phase 9, in miniature: after a build, the gate's
assembly (`build_cached_retriever`) answers every golden query for a
reranker-enabled config with no model at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from groundtruth.chunking.pipeline import chunk_corpus
from groundtruth.config.models import RerankerConfig, RetrievalConfig
from groundtruth.embedding.hashing import embedding_key
from groundtruth.embedding.store import store_dir, write_store
from groundtruth.golden.models import GoldenPair, GoldenSet, Provenance, RelevanceLabel
from groundtruth.rerank.builder import build_rerank_cache
from groundtruth.rerank.cache import RerankCacheMissError
from groundtruth.rerank.protocol import RerankScores
from groundtruth.retrieval.build import build_cached_retriever
from tests.fixtures.mini_corpus import (
    MINI_MODEL_ID,
    MINI_REVISION,
    HashingEmbedder,
    mini_config,
    mini_documents,
)

DOCUMENTS = mini_documents()
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
QUERIES = (
    "what is the standard for granting summary judgment",
    "when is an officer entitled to qualified immunity",
)


class LengthReranker:
    model_id = RERANK_MODEL
    revision = RERANK_REVISION
    build_settings = {"device": "cpu", "torch_threads": 1}  # noqa: RUF012

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, passages: Sequence[str]) -> RerankScores:
        self.calls += 1
        return RerankScores(scores=tuple(float(len(p)) for p in passages), compute_ms=180.0)


def rerank_config(name: str = "mini_rerank") -> RetrievalConfig:
    config = mini_config(name=name, retrieval_mode="hybrid", top_k=3, dense_top_n=10)
    return config.model_copy(
        update={
            "reranker": RerankerConfig(
                enabled=True, model_id=RERANK_MODEL, revision=RERANK_REVISION, top_n_in=5
            )
        }
    )


def golden() -> GoldenSet:
    text = DOCUMENTS[0].text
    needle = "summary judgment. Summary judgment is appropriate only where there is no"
    start = text.index(needle)
    pairs = tuple(
        GoldenPair(
            query_id=f"q-{i}",
            query=query,
            category="factual-lookup",
            labels=(
                RelevanceLabel(
                    doc_id=DOCUMENTS[0].doc_id,
                    char_start=start,
                    char_end=start + len(needle),
                    gain=3,
                    quote=needle,
                ),
            ),
            provenance=Provenance(
                origin="human",
                reviewer="test",
                reviewed_at="2026-01-01T00:00:00+00:00",
                review_action="accepted",
            ),
        )
        for i, query in enumerate(QUERIES)
    )
    return GoldenSet(pairs=pairs)


def committed_embeddings(cache_root: Path, config: RetrievalConfig) -> None:
    """A committed-style embedding store covering every chunk and golden query."""
    texts = sorted({c.text for c in chunk_corpus(DOCUMENTS, config.chunking)} | set(QUERIES))
    vectors = HashingEmbedder().embed(texts)
    keys = [
        embedding_key(t, model_id=MINI_MODEL_ID, revision=MINI_REVISION, normalize=True)
        for t in texts
    ]
    write_store(
        store_dir(cache_root / "embeddings", MINI_MODEL_ID, MINI_REVISION),
        keys,
        np.asarray(vectors, dtype=np.float32),
        model_id=MINI_MODEL_ID,
        revision=MINI_REVISION,
        normalize=True,
    )


class TestBuildRerankCache:
    def test_records_every_golden_query(self, tmp_path: Path):
        report = build_rerank_cache(
            [rerank_config()], DOCUMENTS, HashingEmbedder(), golden(), LengthReranker(), tmp_path
        )
        assert report.queries == len(QUERIES)
        assert report.computed == len(QUERIES)
        assert report.pairs > 0

    def test_a_second_build_reuses_everything(self, tmp_path: Path):
        build_rerank_cache(
            [rerank_config()], DOCUMENTS, HashingEmbedder(), golden(), LengthReranker(), tmp_path
        )
        delegate = LengthReranker()
        report = build_rerank_cache(
            [rerank_config()], DOCUMENTS, HashingEmbedder(), golden(), delegate, tmp_path
        )
        assert delegate.calls == 0
        assert report.reused == len(QUERIES)

    def test_records_the_delegates_build_settings(self, tmp_path: Path):
        report = build_rerank_cache(
            [rerank_config()], DOCUMENTS, HashingEmbedder(), golden(), LengthReranker(), tmp_path
        )
        meta = (report.directory / "meta.json").read_text(encoding="utf-8")
        assert '"torch_threads": 1' in meta

    def test_refuses_a_config_that_does_not_enable_this_reranker(self, tmp_path: Path):
        plain = mini_config(retrieval_mode="hybrid", top_k=3, dense_top_n=10)
        with pytest.raises(ValueError, match="does not enable"):
            build_rerank_cache(
                [plain], DOCUMENTS, HashingEmbedder(), golden(), LengthReranker(), tmp_path
            )

    def test_refuses_an_empty_config_list(self, tmp_path: Path):
        with pytest.raises(ValueError, match="no reranker-enabled"):
            build_rerank_cache(
                [], DOCUMENTS, HashingEmbedder(), golden(), LengthReranker(), tmp_path
            )


class TestBuildCachedRetriever:
    def test_serves_every_golden_query_with_no_model(self, tmp_path: Path):
        config = rerank_config()
        committed_embeddings(tmp_path, config)
        build_rerank_cache(
            [config], DOCUMENTS, HashingEmbedder(), golden(), LengthReranker(), tmp_path / "rerank"
        )

        retriever = build_cached_retriever(config, DOCUMENTS, tmp_path)
        for query in QUERIES:
            result = retriever.retrieve(query)
            assert result.passages
            assert all(p.scores.rerank is not None for p in result.passages)
            # The recorded model cost, not the lookup's (ADR-0012).
            assert result.latency_ms.rerank == 180.0

    def test_a_query_outside_the_cache_is_a_loud_miss(self, tmp_path: Path):
        config = rerank_config()
        committed_embeddings(tmp_path, config)
        build_rerank_cache(
            [config], DOCUMENTS, HashingEmbedder(), golden(), LengthReranker(), tmp_path / "rerank"
        )
        retriever = build_cached_retriever(config, DOCUMENTS, tmp_path)

        # Not in the rerank cache, and also not in the embedding cache -- the
        # embedding miss fires first; either way the gate fails loudly.
        with pytest.raises(Exception, match="not in the committed"):
            retriever.retrieve("a query nobody ever scored")

    def test_a_missing_rerank_cache_is_named(self, tmp_path: Path):
        config = rerank_config()
        committed_embeddings(tmp_path, config)
        with pytest.raises(RerankCacheMissError, match="gt cache rerank"):
            build_cached_retriever(config, DOCUMENTS, tmp_path)

    def test_a_config_without_the_reranker_needs_no_rerank_cache(self, tmp_path: Path):
        config = mini_config(retrieval_mode="hybrid", top_k=3, dense_top_n=10)
        committed_embeddings(tmp_path, config)
        retriever = build_cached_retriever(config, DOCUMENTS, tmp_path)
        assert retriever.reranker is None
