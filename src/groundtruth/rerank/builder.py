"""Building the committed rerank cache (ADR-0012).

The cache is built by running the **real pipeline** over every golden query,
with a recording wrapper around the live model standing in the reranker's
seat. The candidate set recorded is therefore exactly the one the gate will
later ask about -- by construction, not by a second candidate computation
that could drift from the pipeline's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from groundtruth.config.models import RetrievalConfig
from groundtruth.corpus.models import Document
from groundtruth.embedding.protocol import Embedder
from groundtruth.golden.models import GoldenSet
from groundtruth.progress import ProgressCallback
from groundtruth.rerank.cache import (
    RecordingReranker,
    RerankCacheMissError,
    read_rerank_store,
    rerank_store_dir,
    write_rerank_store,
)
from groundtruth.rerank.protocol import Reranker
from groundtruth.retrieval.build import build_retriever


@dataclass(frozen=True)
class RerankBuildReport:
    directory: Path
    queries: int
    computed: int
    reused: int
    pairs: int


def build_rerank_cache(
    configs: Sequence[RetrievalConfig],
    documents: Sequence[Document],
    embedder: Embedder,
    golden: GoldenSet,
    delegate: Reranker,
    cache_root: Path,
    *,
    on_progress: ProgressCallback | None = None,
) -> RerankBuildReport:
    """Record the delegate's scores for every golden query under every config.

    ``configs`` must all enable the reranker with the delegate's model: one
    store per (model, revision), shared across configs, merged on rebuild so
    an unchanged query costs nothing.
    """
    if not configs:
        raise ValueError("no reranker-enabled configs to build a rerank cache for")
    for config in configs:
        settings = config.reranker
        if not settings.enabled or (settings.model_id, settings.revision) != (
            delegate.model_id,
            delegate.revision,
        ):
            raise ValueError(
                f"config {config.name!r} does not enable reranker "
                f"{delegate.model_id}@{delegate.revision[:12]}"
            )

    directory = rerank_store_dir(cache_root, delegate.model_id, delegate.revision)
    try:
        existing = read_rerank_store(directory)
    except RerankCacheMissError:
        existing = None

    recorder = RecordingReranker(delegate, existing)
    total = len(configs) * len(golden.pairs)
    done = 0
    for config in configs:
        retriever = build_retriever(config, documents, embedder, reranker=recorder)
        for pair in golden.pairs:
            retriever.retrieve(pair.query)
            done += 1
            if on_progress is not None:
                on_progress(done, total)

    store = recorder.store()
    build_settings: dict[str, Any] = dict(getattr(delegate, "build_settings", {}))
    write_rerank_store(directory, store, build_settings=build_settings)

    return RerankBuildReport(
        directory=directory,
        queries=len(store.queries),
        computed=recorder.computed_queries,
        reused=recorder.reused_queries,
        pairs=store.pair_count,
    )
