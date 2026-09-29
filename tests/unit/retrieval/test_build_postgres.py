"""Service-path assembly logic that needs no database."""

from __future__ import annotations

from groundtruth.index.postgres import index_key
from groundtruth.retrieval.build_postgres import served_config
from tests.fixtures.mini_corpus import mini_config


class TestServedConfig:
    def test_a_dense_config_is_served_unchanged(self):
        config = mini_config(retrieval_mode="dense")
        assert served_config(config) == config

    def test_a_bm25_hybrid_is_served_as_pg_fts_under_its_own_name(self):
        # ADR-0007: a result must never claim BM25 ranked it when pg_fts did.
        config = mini_config(name="mini_hybrid", retrieval_mode="hybrid")
        served = served_config(config)
        assert served.lexical is not None
        assert served.lexical.backend == "pg_fts"
        assert served.name == "mini_hybrid+pg_fts"

    def test_the_served_variant_has_its_own_content_hash(self):
        # Different lexical backend is a different experiment; a shared hash
        # would let a served result be mistaken for the gated one.
        config = mini_config(retrieval_mode="hybrid")
        assert served_config(config).config_hash != config.config_hash

    def test_a_config_already_on_pg_fts_is_left_alone(self):
        config = served_config(mini_config(name="x", retrieval_mode="hybrid"))
        assert served_config(config) == config


class TestIndexKey:
    def test_is_stable(self):
        config = mini_config()
        assert index_key(config.chunking, config.embedding) == index_key(
            config.chunking, config.embedding
        )

    def test_changes_with_the_chunking(self):
        # Otherwise rows built under 512-token windows would be served to a
        # 256-token config -- spans that do not match their text.
        config = mini_config()
        other = config.chunking.model_copy(update={"chunk_size": 64, "chunk_overlap": 8})
        assert index_key(config.chunking, config.embedding) != index_key(other, config.embedding)

    def test_ignores_retrieval_settings_that_do_not_change_the_rows(self):
        # top_k and the lexical arm do not change what is stored, so dense_512,
        # hybrid_512 and hybrid_512_rerank share one loaded index.
        dense = mini_config(retrieval_mode="dense", top_k=3)
        hybrid = mini_config(retrieval_mode="hybrid", top_k=5)
        assert index_key(dense.chunking, dense.embedding) == index_key(
            hybrid.chunking, hybrid.embedding
        )
