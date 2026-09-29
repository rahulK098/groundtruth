"""``gt db`` -- load the committed corpus and vectors into the service's Postgres.

Needs the ``api`` extra and a running database (`docker compose up -d db`),
but NO model: every chunk vector comes from the committed embedding cache, so
the rows loaded here are the very vectors the gate scores against. That is
half of what makes the parity test meaningful.
"""

from __future__ import annotations

import numpy as np
import typer

from groundtruth.chunking.pipeline import chunk_corpus
from groundtruth.config.registry import load_all_configs
from groundtruth.corpus.snapshot import verify_snapshot
from groundtruth.embedding.cache import load_cached_only_embedder
from groundtruth.paths import cache_dir, configs_dir, corpus_dir
from groundtruth.settings import Settings

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("load")
def load() -> None:
    """Load every configuration's chunks and committed vectors. Idempotent."""
    from groundtruth.index.postgres import (
        ApiExtraNotInstalledError,
        PostgresIndexError,
        connect,
        ensure_schema,
        index_key,
        load_index,
    )

    documents = verify_snapshot(corpus_dir())
    configs = load_all_configs(configs_dir())
    spaces = {(cfg.chunking, cfg.embedding) for cfg in configs.values()}

    try:
        conn = connect(Settings().gt_database_url)
    except (ApiExtraNotInstalledError, PostgresIndexError) as exc:
        typer.echo(f"FAIL: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    with conn:
        for chunking, embedding in sorted(spaces, key=lambda s: -s[0].chunk_size):
            ensure_schema(conn, embedding.dimension)
            chunks = chunk_corpus(documents, chunking)
            embedder = load_cached_only_embedder(
                embedding.model_id, embedding.revision, cache_dir() / "embeddings"
            )
            vectors = np.asarray(embedder.embed([c.text for c in chunks]), dtype=np.float32)
            key = index_key(chunking, embedding)
            count = load_index(conn, key, chunks, vectors)
            typer.echo(f"loaded {count:>6,} chunks  chunk_size={chunking.chunk_size:<4}  key {key}")


@app.command("status")
def status() -> None:
    """Show which indexes are loaded and how many rows each holds."""
    from groundtruth.index.postgres import (
        ApiExtraNotInstalledError,
        PostgresIndexError,
        connect,
        loaded_keys,
    )

    try:
        conn = connect(Settings().gt_database_url)
    except (ApiExtraNotInstalledError, PostgresIndexError) as exc:
        typer.echo(f"FAIL: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    with conn:
        keys = loaded_keys(conn)
    if not keys:
        typer.echo("no indexes loaded; run `uv run gt db load`")
        raise typer.Exit(code=1)
    for key, count in sorted(keys.items()):
        typer.echo(f"{key}  {count:,} rows")
