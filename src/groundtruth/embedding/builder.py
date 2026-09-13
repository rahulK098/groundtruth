"""Building the committed embedding cache.

One store per (model, revision), shared across chunking configurations. The
512- and 256-token configs produce different chunks but many identical
strings, and because cache keys address the exact text, those collapse to one
vector automatically.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from groundtruth.chunking.pipeline import chunk_corpus
from groundtruth.config.models import ChunkingConfig, EmbeddingConfig
from groundtruth.corpus.models import Document
from groundtruth.embedding.cache import CachedEmbedder
from groundtruth.embedding.hashing import embedding_key
from groundtruth.embedding.protocol import Embedder
from groundtruth.embedding.store import (
    EmbeddingStoreError,
    read_store,
    store_dir,
    write_store,
)
from groundtruth.progress import ProgressCallback


@dataclass(frozen=True)
class BuildReport:
    total_strings: int
    unique_strings: int
    reused: int
    computed: int
    directory: Path


def collect_texts(documents: Sequence[Document], chunkings: Iterable[ChunkingConfig]) -> list[str]:
    """Every distinct string that must be embeddable, in deterministic order.

    Sorted, not insertion-ordered: the cache is committed, so a stable order
    keeps its diff meaningful across rebuilds.
    """
    texts: set[str] = set()
    for chunking in chunkings:
        for chunk in chunk_corpus(documents, chunking):
            texts.add(chunk.text)
    return sorted(texts)


def build_cache(
    documents: Sequence[Document],
    chunkings: Sequence[ChunkingConfig],
    embedding: EmbeddingConfig,
    delegate: Embedder,
    cache_root: Path,
    *,
    batch: int = 256,
    on_progress: ProgressCallback | None = None,
) -> BuildReport:
    """Embed every chunk string and write the committed cache."""
    directory = store_dir(cache_root, embedding.model_id, embedding.revision)

    try:
        existing = read_store(directory)
    except EmbeddingStoreError:
        existing = None

    texts = collect_texts(documents, chunkings)
    cached = CachedEmbedder(existing, delegate)

    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch):
        window = texts[start : start + batch]
        vectors.append(cached.embed(window))
        if on_progress is not None:
            on_progress(min(start + batch, len(texts)), len(texts))

    matrix = np.vstack(vectors) if vectors else np.zeros((0, delegate.dimension), dtype=np.float32)
    keys = [
        embedding_key(
            text,
            model_id=embedding.model_id,
            revision=embedding.revision,
            normalize=embedding.normalize,
        )
        for text in texts
    ]

    build_settings = getattr(delegate, "build_settings", {})
    write_store(
        directory,
        keys,
        matrix,
        model_id=embedding.model_id,
        revision=embedding.revision,
        normalize=embedding.normalize,
        build_settings=dict(build_settings),
    )

    computed = len(cached.computed)
    return BuildReport(
        total_strings=len(texts),
        unique_strings=len(set(keys)),
        reused=len(texts) - computed,
        computed=computed,
        directory=directory,
    )
