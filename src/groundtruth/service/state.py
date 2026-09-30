"""What the service serves from, built once at startup.

``ServiceState`` is plain data: configs, the retrievers built for the served
subset, and the identity ``/version`` reports. ``build_state`` constructs the
production one -- Postgres dense arm, in-process BM25 (ADR-0014), and a query
embedder that serves every committed vector from the cache and computes only
the one vector an arbitrary query needs. Tests hand ``create_app`` a state
built over the mini corpus instead, so the HTTP layer is testable with no
model and no database.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from groundtruth import __version__
from groundtruth.config.models import RetrievalConfig
from groundtruth.retrieval.pipeline import Retriever
from groundtruth.service.schemas import ModelIdentity, VersionInfo

#: Served unless GT_SERVE_CONFIGS says otherwise. The reranker arm is
#: excluded by default: an arbitrary query is not in its committed score
#: cache, so the live cross-encoder would run at ~105 s per query on CPU.
DEFAULT_SERVED: Final[tuple[str, ...]] = ("dense_512", "dense_256", "hybrid_512")


@dataclass(frozen=True)
class ServiceState:
    configs: Mapping[str, RetrievalConfig]
    retrievers: Mapping[str, Retriever]
    version: VersionInfo

    @property
    def ready(self) -> bool:
        return bool(self.retrievers)


def git_sha() -> str:
    """Short SHA of the running checkout, or ``unknown`` outside a git tree."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def version_info(
    configs: Mapping[str, RetrievalConfig], *, corpus_manifest_hash: str, golden_set_hash: str
) -> VersionInfo:
    embeddings = {(c.embedding.model_id, c.embedding.revision) for c in configs.values()}
    rerankers = {
        (c.reranker.model_id, c.reranker.revision)
        for c in configs.values()
        if c.reranker.enabled and c.reranker.model_id and c.reranker.revision
    }
    embedding_id, embedding_rev = sorted(embeddings)[0]
    reranker = None
    if rerankers:
        rerank_id, rerank_rev = sorted(rerankers)[0]
        reranker = ModelIdentity(id=str(rerank_id), revision=str(rerank_rev))
    return VersionInfo(
        git_sha=git_sha(),
        code_version=__version__,
        embedding_model=ModelIdentity(id=embedding_id, revision=embedding_rev),
        reranker_model=reranker,
        corpus_manifest_hash=corpus_manifest_hash,
        golden_set_hash=golden_set_hash,
    )


def served_names(requested: str, available: Mapping[str, Any]) -> tuple[str, ...]:
    """Parse GT_SERVE_CONFIGS, refusing a name that is not a shipped config."""
    names = tuple(n.strip() for n in requested.split(",") if n.strip()) or DEFAULT_SERVED
    unknown = [n for n in names if n not in available]
    if unknown:
        raise ValueError(f"GT_SERVE_CONFIGS names unknown configs: {', '.join(unknown)}")
    return names


def build_state() -> ServiceState:  # pragma: no cover - needs Postgres and the models extra
    """The production state: Postgres + committed caches + a live query embedder."""
    from groundtruth.config.registry import load_all_configs
    from groundtruth.corpus.snapshot import read_manifest, verify_snapshot
    from groundtruth.embedding.cache import CachedEmbedder
    from groundtruth.embedding.sentence_transformer import SentenceTransformerEmbedder
    from groundtruth.embedding.store import read_store, store_dir
    from groundtruth.golden.store import read_golden_set
    from groundtruth.index.postgres import connect
    from groundtruth.paths import cache_dir, configs_dir, corpus_dir, golden_dir
    from groundtruth.rerank.cache import RecordingReranker, read_rerank_store, rerank_store_dir
    from groundtruth.rerank.cross_encoder import CrossEncoderReranker
    from groundtruth.retrieval.build_postgres import build_postgres_retriever
    from groundtruth.settings import Settings

    settings = Settings()
    configs = dict(load_all_configs(configs_dir()))
    documents = verify_snapshot(corpus_dir())
    conn = connect(settings.gt_database_url)

    embedders: dict[tuple[str, str], CachedEmbedder] = {}
    retrievers: dict[str, Retriever] = {}
    for name in served_names(settings.gt_serve_configs, configs):
        config = configs[name]
        key = (config.embedding.model_id, config.embedding.revision)
        if key not in embedders:
            embedders[key] = CachedEmbedder(
                read_store(store_dir(cache_dir() / "embeddings", *key)),
                SentenceTransformerEmbedder(*key, normalize=config.embedding.normalize),
            )

        reranker = None
        settings_rr = config.reranker
        if settings_rr.enabled and settings_rr.model_id and settings_rr.revision:
            # Committed scores first; the live model only for what they miss.
            directory = rerank_store_dir(
                cache_dir() / "rerank", settings_rr.model_id, settings_rr.revision
            )
            reranker = RecordingReranker(
                CrossEncoderReranker(settings_rr.model_id, settings_rr.revision),
                read_rerank_store(directory),
            )

        retrievers[name] = build_postgres_retriever(
            config, documents, embedders[key], conn, reranker=reranker
        )

    return ServiceState(
        configs=configs,
        retrievers=retrievers,
        version=version_info(
            configs,
            corpus_manifest_hash=read_manifest(corpus_dir()).manifest_hash,
            golden_set_hash=read_golden_set(golden_dir()).golden_set_hash,
        ),
    )
