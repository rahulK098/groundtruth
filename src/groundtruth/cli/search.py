"""``gt search`` -- run one query through a retrieval configuration.

A human affordance, not part of the gate. It needs the ``models`` extra,
because an arbitrary query typed at a terminal is by definition not in the
committed cache, and a query vector has to come from somewhere.

Note which embedder it uses: ``CachedEmbedder``, not ``CachedOnlyEmbedder``.
Every passage vector still comes from the committed cache -- only the query is
computed. The gate uses the cache-only variant, which cannot compute anything
at all (ADR-0003), and nothing here changes that.
"""

from __future__ import annotations

from pathlib import Path

import typer

from groundtruth.config.models import RetrievalConfig
from groundtruth.config.registry import load_config, shipped_configs_dir
from groundtruth.corpus.snapshot import verify_snapshot
from groundtruth.embedding.store import read_store, store_dir
from groundtruth.paths import cache_dir, corpus_dir
from groundtruth.retrieval.models import RetrievalResult

#: Enough of a passage to recognise it, short enough to scan a top-10.
_SNIPPET_CHARS = 220


def render(result: RetrievalResult, config: RetrievalConfig) -> str:
    """Format a result for a terminal.

    Separate from the command so it is testable without a model: the thing
    most likely to be wrong here is the formatting, not the plumbing.
    """
    lines = [
        f"query  : {result.query}",
        f"config : {config.name}  ({config.retrieval_mode}, hash {config.config_hash})",
        (
            f"latency: embed {result.latency_ms.embed:.0f} ms  "
            f"dense {result.latency_ms.dense:.0f} ms  "
            f"lexical {result.latency_ms.lexical:.0f} ms  "
            f"fuse {result.latency_ms.fuse:.0f} ms  "
            f"total {result.latency_ms.total:.0f} ms"
        ),
        "",
    ]

    if not result.passages:
        lines.append("no passages retrieved")
        return "\n".join(lines)

    for passage in result.passages:
        scores = passage.scores
        # Every stage is shown separately. "the dense arm never surfaced it"
        # and "fusion buried it" are different failures, and a single blended
        # number cannot tell them apart.
        parts = [
            f"{label} {value:.4f}"
            for label, value in (
                ("dense", scores.dense),
                ("lexical", scores.lexical),
                ("fused", scores.fused),
                ("rerank", scores.rerank),
            )
            if value is not None
        ]
        snippet = " ".join(passage.text[:_SNIPPET_CHARS].split())
        ellipsis = "..." if len(passage.text) > _SNIPPET_CHARS else ""
        lines.append(
            f"{passage.rank:2d}. {passage.doc_id}  "
            f"[{passage.char_start}:{passage.char_end}]  {'  '.join(parts)}"
        )
        lines.append(f"    {snippet}{ellipsis}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def search(
    query: str = typer.Argument(..., help="The query to run."),
    config_name: str = typer.Option("hybrid_512", "--config", help="Configuration to use."),
    top_k: int | None = typer.Option(None, help="Override the config's top_k."),
    configs: Path | None = typer.Option(None, help="Config directory. Defaults to configs/."),
) -> None:
    """Retrieve passages for one query. Requires the `models` extra."""
    # Imported here so `gt --help` still works in the gate environment, where
    # torch is deliberately absent.
    from groundtruth.embedding.cache import CachedEmbedder
    from groundtruth.embedding.sentence_transformer import (
        ModelExtraNotInstalledError,
        SentenceTransformerEmbedder,
    )
    from groundtruth.retrieval.build import build_retriever

    directory = configs or shipped_configs_dir()
    config = load_config(directory / f"{config_name}.yaml")

    try:
        delegate = SentenceTransformerEmbedder(
            config.embedding.model_id,
            config.embedding.revision,
            normalize=config.embedding.normalize,
        )
    except ModelExtraNotInstalledError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    store = read_store(
        store_dir(cache_dir() / "embeddings", config.embedding.model_id, config.embedding.revision)
    )
    retriever = build_retriever(
        config, verify_snapshot(corpus_dir()), CachedEmbedder(store, delegate)
    )
    typer.echo(render(retriever.retrieve(query, top_k=top_k), config))
