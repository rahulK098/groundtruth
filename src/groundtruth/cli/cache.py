"""``gt cache`` commands."""

from __future__ import annotations

from pathlib import Path

import typer

from groundtruth.config.registry import load_all_configs
from groundtruth.corpus.snapshot import verify_snapshot
from groundtruth.embedding.cache import load_cached_only_embedder
from groundtruth.embedding.store import EmbeddingStoreError, store_dir, verify_store
from groundtruth.golden.store import read_golden_set
from groundtruth.paths import cache_dir, configs_dir, corpus_dir, golden_dir

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _progress(done: int, total: int) -> None:
    typer.echo(f"  ... {done}/{total}", err=True)


@app.command("build")
def build(
    batch: int = typer.Option(256, help="Strings per write batch."),
    encode_batch: int = typer.Option(32, help="Strings per model forward pass."),
) -> None:
    """Embed every chunk string and write the committed cache.

    Requires the `models` extra and downloads ~133 MB of weights on first run.
    This is the ONLY command that needs them -- the gate serves committed
    vectors and cannot compute.
    """
    # Imported here, not at module scope: this module is loaded by `gt --help`
    # in the gate environment, where torch is deliberately absent.
    from groundtruth.embedding.builder import build_cache
    from groundtruth.embedding.sentence_transformer import (
        ModelExtraNotInstalledError,
        SentenceTransformerEmbedder,
    )

    documents = verify_snapshot(corpus_dir())
    configs = load_all_configs(configs_dir())

    chunkings = sorted(
        {cfg.chunking for cfg in configs.values()}, key=lambda c: c.chunk_size, reverse=True
    )
    embeddings = {cfg.embedding for cfg in configs.values()}
    if len(embeddings) != 1:
        typer.echo(
            f"Expected one embedding model across all configs, found {len(embeddings)}. "
            "Building multiple stores is not supported yet.",
            err=True,
        )
        raise typer.Exit(code=1)
    embedding = next(iter(embeddings))

    # Golden-set queries join the same store as the chunks, with their prefix
    # already applied -- the cache key hashes the exact string sent to the
    # model (ADR-0003), so this is what lets the gate answer every golden
    # query through CachedOnlyEmbedder without ever touching the network.
    # Missing or empty is fine: an early Phase 5 checkout has no golden set
    # yet, and this command still has chunks to embed.
    golden = read_golden_set(golden_dir())
    extra_texts = {embedding.query_prefix + pair.query for pair in golden.pairs}

    typer.echo(
        f"corpus     : {len(documents)} documents\n"
        f"chunkings  : {', '.join(str(c.chunk_size) for c in chunkings)}\n"
        f"model      : {embedding.model_id}@{embedding.revision[:12]}\n"
        f"golden set : {len(golden.pairs)} queries"
    )

    try:
        delegate = SentenceTransformerEmbedder(
            embedding.model_id,
            embedding.revision,
            normalize=embedding.normalize,
            batch_size=encode_batch,
        )
    except ModelExtraNotInstalledError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if delegate.dimension != embedding.dimension:
        typer.echo(
            f"Model reports dimension {delegate.dimension} but the config "
            f"declares {embedding.dimension}.",
            err=True,
        )
        raise typer.Exit(code=1)

    report = build_cache(
        documents,
        chunkings,
        embedding,
        delegate,
        cache_dir() / "embeddings",
        batch=batch,
        extra_texts=extra_texts,
        on_progress=_progress,
    )

    typer.echo(
        f"\nwrote {report.total_strings:,} vectors to {report.directory}\n"
        f"  computed : {report.computed:,}\n"
        f"  reused   : {report.reused:,}"
    )


@app.command("rerank")
def rerank(
    encode_batch: int = typer.Option(16, help="Pairs per cross-encoder forward pass."),
) -> None:
    """Score every golden query's candidates with the cross-encoder and commit them.

    Requires the `models` extra and downloads ~1.1 GB of reranker weights on
    first run. Like `gt cache build`, this is the only place the model runs:
    the gate serves the committed scores and cannot compute (ADR-0012).

    Run after `gt cache build` -- the rerank candidates come from the real
    pipeline, which needs every golden query's embedding already committed.
    """
    from groundtruth.embedding.sentence_transformer import ModelExtraNotInstalledError
    from groundtruth.rerank.builder import build_rerank_cache
    from groundtruth.rerank.cross_encoder import CrossEncoderReranker

    documents = verify_snapshot(corpus_dir())
    configs = [cfg for cfg in load_all_configs(configs_dir()).values() if cfg.reranker.enabled]
    if not configs:
        typer.echo("no configuration enables the reranker; nothing to build")
        return

    models = {(cfg.reranker.model_id, cfg.reranker.revision) for cfg in configs}
    embeddings = {cfg.embedding for cfg in configs}
    if len(models) != 1 or len(embeddings) != 1:
        typer.echo(
            "Expected one reranker model and one embedding model across reranker "
            "configs; building multiple stores is not supported yet.",
            err=True,
        )
        raise typer.Exit(code=1)
    model_id, revision = next(iter(models))
    embedding = next(iter(embeddings))
    if model_id is None or revision is None:  # pragma: no cover - RerankerConfig forbids this
        raise typer.Exit(code=1)

    golden = read_golden_set(golden_dir())
    if not golden.pairs:
        typer.echo("the golden set is empty; there are no queries to rerank", err=True)
        raise typer.Exit(code=1)

    typer.echo(
        f"configs    : {', '.join(cfg.name for cfg in configs)}\n"
        f"model      : {model_id}@{revision[:12]}\n"
        f"golden set : {len(golden.pairs)} queries"
    )

    try:
        delegate = CrossEncoderReranker(model_id, revision, batch_size=encode_batch)
    except ModelExtraNotInstalledError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    embedder = load_cached_only_embedder(
        embedding.model_id, embedding.revision, cache_dir() / "embeddings"
    )
    report = build_rerank_cache(
        configs,
        documents,
        embedder,
        golden,
        delegate,
        cache_dir() / "rerank",
        on_progress=_progress,
    )

    typer.echo(
        f"\nwrote {report.pairs:,} scores for {report.queries} queries to {report.directory}\n"
        f"  computed : {report.computed}\n"
        f"  reused   : {report.reused}"
    )


@app.command("verify")
def verify(
    root: Path | None = typer.Option(None, help="Cache root. Defaults to data/cache."),
) -> None:
    """Check the committed cache is internally consistent.

    Catches keys and vectors drifting out of alignment, which would serve the
    wrong embedding for every row past the break -- plausible numbers, silently
    wrong, and invisible to every other check.
    """
    base = (root or cache_dir()) / "embeddings"
    configs = load_all_configs(configs_dir())
    embeddings = {cfg.embedding for cfg in configs.values()}

    failures = 0
    for embedding in sorted(embeddings, key=lambda e: e.model_id):
        directory = store_dir(base, embedding.model_id, embedding.revision)
        try:
            count = verify_store(directory)
        except EmbeddingStoreError as exc:
            typer.echo(f"FAIL {embedding.model_id}: {exc}", err=True)
            failures += 1
            continue
        typer.echo(f"OK   {embedding.model_id}@{embedding.revision[:12]}: {count:,} vectors")

    if failures:
        raise typer.Exit(code=1)
