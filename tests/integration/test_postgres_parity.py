"""The parity test: the claim that licenses calling two paths one system.

ADR-0001: "An integration parity test asserts the Postgres path returns an
identical top-10 to the NumPy path. That test is what licenses describing
them as one system."

Runs against the real corpus, the real golden set and the real committed
vectors, loaded into Postgres by `gt db load`. Needs `docker compose up -d
db` and the `api` extra; skips cleanly without them, so the gate and the fast
path never require infrastructure (ADR-0006).

Parity is asserted for DENSE configs only, and that is not a gap: the
service's lexical arm is pg_fts, which is deliberately not BM25 (ADR-0007),
so a hybrid config is served as a differently-named, differently-hashed
variant. What IS asserted for hybrid is that the variant is labelled honestly.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from groundtruth.config.models import RetrievalConfig
from groundtruth.config.registry import load_config, shipped_configs_dir
from groundtruth.corpus.models import Document
from groundtruth.corpus.snapshot import verify_snapshot
from groundtruth.embedding.cache import load_cached_only_embedder
from groundtruth.golden.models import GoldenSet
from groundtruth.golden.store import read_golden_set
from groundtruth.paths import cache_dir, corpus_dir, golden_dir
from groundtruth.retrieval.build import build_cached_retriever
from groundtruth.settings import Settings

pytestmark = pytest.mark.integration

DENSE_CONFIGS = ("dense_512", "dense_256")

#: float32 rounding, not a tolerance for disagreement: both paths score the
#: same unit vectors, and observed differences are ~1e-7.
SCORE_TOLERANCE = 1e-5


@pytest.fixture(scope="module")
def conn() -> Iterator[Any]:
    try:
        from groundtruth.index.postgres import (
            ApiExtraNotInstalledError,
            PostgresIndexError,
            connect,
            loaded_keys,
        )
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"api extra not installed: {exc}")

    try:
        connection = connect(Settings().gt_database_url)
    except (ApiExtraNotInstalledError, PostgresIndexError) as exc:
        pytest.skip(f"Postgres unavailable: {exc}")

    if not loaded_keys(connection):
        connection.close()
        pytest.skip("Postgres has no loaded indexes; run `uv run gt db load`")

    yield connection
    connection.close()


@pytest.fixture(scope="module")
def documents() -> tuple[Document, ...]:
    return verify_snapshot(corpus_dir())


@pytest.fixture(scope="module")
def golden() -> GoldenSet:
    return read_golden_set(golden_dir())


def _config(name: str) -> RetrievalConfig:
    return load_config(shipped_configs_dir() / f"{name}.yaml")


def _postgres_retriever(config: RetrievalConfig, documents: tuple[Document, ...], conn: Any):
    from groundtruth.retrieval.build_postgres import build_postgres_retriever

    embedder = load_cached_only_embedder(
        config.embedding.model_id, config.embedding.revision, cache_dir() / "embeddings"
    )
    return build_postgres_retriever(config, documents, embedder, conn)


@pytest.mark.parametrize("name", DENSE_CONFIGS)
def test_postgres_returns_the_identical_top_10_for_every_golden_query(
    name: str, conn: Any, documents: tuple[Document, ...], golden: GoldenSet
) -> None:
    config = _config(name)
    in_process = build_cached_retriever(config, documents, cache_dir())
    postgres = _postgres_retriever(config, documents, conn)

    mismatched = [
        pair.query_id
        for pair in golden.pairs
        if [p.chunk_id for p in in_process.retrieve(pair.query).passages]
        != [p.chunk_id for p in postgres.retrieve(pair.query).passages]
    ]
    assert not mismatched, (
        f"{name}: {len(mismatched)} of {len(golden.pairs)} golden queries rank differently "
        f"on Postgres (first: {mismatched[:5]}). The two paths are no longer one system."
    )


@pytest.mark.parametrize("name", DENSE_CONFIGS)
def test_dense_scores_agree_to_float32_rounding(
    name: str, conn: Any, documents: tuple[Document, ...], golden: GoldenSet
) -> None:
    config = _config(name)
    in_process = build_cached_retriever(config, documents, cache_dir())
    postgres = _postgres_retriever(config, documents, conn)

    worst = 0.0
    for pair in golden.pairs[:20]:
        for left, right in zip(
            in_process.retrieve(pair.query).passages,
            postgres.retrieve(pair.query).passages,
            strict=True,
        ):
            assert left.scores.dense is not None and right.scores.dense is not None
            worst = max(worst, abs(left.scores.dense - right.scores.dense))
    assert worst < SCORE_TOLERANCE


def test_the_served_hybrid_is_labelled_as_pg_fts(
    conn: Any, documents: tuple[Document, ...], golden: GoldenSet
) -> None:
    postgres = _postgres_retriever(_config("hybrid_512"), documents, conn)
    # A golden query: the cache-only embedder (correctly) refuses any string
    # that is not committed, so an ad-hoc query here would be a cache miss.
    result = postgres.retrieve(golden.pairs[0].query)

    assert result.config_name == "hybrid_512+pg_fts"
    assert result.config_hash != _config("hybrid_512").config_hash
    assert postgres.lexical_index is not None
    assert postgres.lexical_index.backend == "pg_fts"


def test_pg_fts_only_returns_chunks_matching_a_query_term(
    conn: Any, documents: tuple[Document, ...]
) -> None:
    postgres = _postgres_retriever(_config("hybrid_512"), documents, conn)
    assert postgres.lexical_index is not None
    hits = postgres.lexical_index.search("certiorari", 20)
    assert hits
    chunks = {c.chunk_id: c for c in postgres._chunks.values()}
    assert all("certiorari" in chunks[hit.chunk_id].text.lower() for hit in hits)


def test_pg_fts_matches_any_term_not_all_terms(conn: Any, documents: tuple[Document, ...]) -> None:
    # plainto_tsquery ANDs terms; an AND over a long natural-language question
    # matches almost nothing. The index rewrites it to OR, the same candidate
    # rule the in-process BM25 arm uses.
    postgres = _postgres_retriever(_config("hybrid_512"), documents, conn)
    assert postgres.lexical_index is not None
    assert postgres.lexical_index.search("certiorari zzzznotawordzzzz", 5)


def test_reloading_is_idempotent(conn: Any, documents: tuple[Document, ...]) -> None:
    from groundtruth.index.postgres import loaded_keys

    before = loaded_keys(conn)
    postgres = _postgres_retriever(_config("dense_512"), documents, conn)
    assert postgres.vector_index.count in before.values()
