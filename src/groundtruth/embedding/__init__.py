"""Embedding: cache-first, with the model libraries behind an optional extra."""

from __future__ import annotations

from groundtruth.embedding.cache import (
    CachedEmbedder,
    CachedOnlyEmbedder,
    EmbeddingCacheMissError,
)
from groundtruth.embedding.hashing import embedding_key, text_digest
from groundtruth.embedding.protocol import Embedder
from groundtruth.embedding.store import (
    EmbeddingStoreError,
    LoadedStore,
    read_store,
    store_dir,
    verify_store,
    write_store,
)

__all__ = [
    "CachedEmbedder",
    "CachedOnlyEmbedder",
    "Embedder",
    "EmbeddingCacheMissError",
    "EmbeddingStoreError",
    "LoadedStore",
    "embedding_key",
    "read_store",
    "store_dir",
    "text_digest",
    "verify_store",
    "write_store",
]
