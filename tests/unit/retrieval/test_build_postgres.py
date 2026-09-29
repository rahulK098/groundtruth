"""Service-path assembly logic that needs no database."""

from __future__ import annotations

from groundtruth.index.postgres import index_key
from tests.fixtures.mini_corpus import mini_config


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
