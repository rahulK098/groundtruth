"""``gt run`` and ``gt baseline bless`` -- the evaluation path's CLI surface.

Both commands serve everything from committed caches -- ``CachedOnlyEmbedder``
for vectors, ``CachedOnlyReranker`` for cross-encoder scores. Unlike
``gt search``, neither ever needs the ``models`` extra: every string and
every (query, passage) pair an evaluation run could need is already committed
(ADR-0003, ADR-0012), and a miss here is a bug to fix by regenerating a
cache, never a reason to fall back to a live model.
"""

from __future__ import annotations

from pathlib import Path

import typer

from groundtruth.config.models import RetrievalConfig
from groundtruth.config.registry import (
    ConfigError,
    load_all_configs,
    load_config,
    shipped_configs_dir,
)
from groundtruth.corpus.models import Document
from groundtruth.corpus.snapshot import SnapshotError, verify_snapshot
from groundtruth.embedding.cache import EmbeddingCacheMissError
from groundtruth.golden.models import GoldenSet
from groundtruth.golden.store import GoldenStoreError, read_golden_set
from groundtruth.paths import cache_dir, corpus_dir, golden_dir
from groundtruth.rerank.cache import RerankCacheMissError
from groundtruth.retrieval.build import UnsupportedBackendError, build_cached_retriever
from groundtruth.retrieval.pipeline import RetrieverError
from groundtruth.scoring.evaluate import score_config
from groundtruth.scoring.models import ScoringReport
from groundtruth.scoring.report_io import ReportIOError, read_report, write_baseline, write_run

#: Wired directly onto ``gt run`` by main.py -- not a sub-Typer of its own,
#: since `gt run` takes no subcommand.
baseline_app = typer.Typer(no_args_is_help=True, add_completion=False)


def _load_golden(directory: Path | None) -> GoldenSet:
    try:
        return read_golden_set(directory or golden_dir())
    except GoldenStoreError as exc:
        typer.echo(f"FAIL: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _run_one(
    config: RetrievalConfig,
    golden: GoldenSet,
    *,
    corpus_manifest_hash: str,
    documents: tuple[Document, ...],
    cache_root: Path,
    code_version: str,
) -> ScoringReport:
    """Score one config from the committed caches, or fail loudly naming why.

    A cache miss -- embedding or rerank -- can surface while building the
    retriever or only once a particular query asks for it, so both are
    inside the guard.
    """
    try:
        retriever = build_cached_retriever(config, documents, cache_root)
        return score_config(
            retriever, golden, corpus_manifest_hash=corpus_manifest_hash, code_version=code_version
        )
    except (
        RetrieverError,
        UnsupportedBackendError,
        EmbeddingCacheMissError,
        RerankCacheMissError,
    ) as exc:
        typer.echo(f"{config.name}: FAIL -- {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _summary_line(report: ScoringReport) -> str:
    recall = report.overall.get("recall", {}).get(10)
    ndcg = report.overall.get("ndcg", {}).get(10)
    mrr = report.overall.get("mrr", {}).get(10)
    parts = [f"{report.config_name:<20}"]
    if recall is not None:
        parts.append(f"recall@10 {recall.mean:.3f}")
    if mrr is not None:
        parts.append(f"mrr@10 {mrr.mean:.3f}")
    if ndcg is not None:
        parts.append(f"ndcg@10 {ndcg.mean:.3f}")
    parts.append(f"latency {report.latency.total_ms:.2f}ms")
    if report.n_skipped:
        parts.append(f"skipped {report.n_skipped}")
    return "  ".join(parts)


def run(
    config_name: str = typer.Option("", "--config", help="Run one config by name."),
    all_configs: bool = typer.Option(False, "--all", help="Run every config."),
    configs: Path | None = typer.Option(None, help="Config directory. Defaults to configs/."),
    corpus: Path | None = typer.Option(
        None, help="Corpus snapshot directory. Defaults to data/corpus."
    ),
    cache: Path | None = typer.Option(None, help="Embedding cache root. Defaults to data/cache."),
    golden: Path | None = typer.Option(None, help="Golden-set directory. Defaults to data/golden."),
    out: Path = typer.Option(Path("results"), "--out", help="Results directory."),
) -> None:
    """Score one or every retrieval configuration against the golden set.

    Needs no model, no network, no GPU -- every string is already in the
    committed embedding cache. This is the command the gate itself calls, so
    a run that succeeds here is one the gate can trust.
    """
    if bool(config_name) == all_configs:
        typer.echo("pass exactly one of --config NAME or --all", err=True)
        raise typer.Exit(code=1)

    config_dir = configs or shipped_configs_dir()
    try:
        available = (
            load_all_configs(config_dir)
            if all_configs
            else {config_name: load_config(config_dir / f"{config_name}.yaml")}
        )
    except ConfigError as exc:
        typer.echo(f"FAIL: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        documents = verify_snapshot(corpus or corpus_dir())
    except SnapshotError as exc:
        typer.echo(f"FAIL: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    golden_set = _load_golden(golden)
    from groundtruth import __version__
    from groundtruth.corpus.snapshot import read_manifest

    manifest_hash = read_manifest(corpus or corpus_dir()).manifest_hash
    cache_root = cache or cache_dir()

    typer.echo(f"golden set : {len(golden_set.pairs)} pairs  {golden_set.golden_set_hash}")
    typer.echo(f"corpus     : {len(documents)} documents  {manifest_hash}\n")

    for name in sorted(available):
        report = _run_one(
            available[name],
            golden_set,
            corpus_manifest_hash=manifest_hash,
            documents=documents,
            cache_root=cache_root,
            code_version=__version__,
        )
        path = write_run(report, out)
        typer.echo(_summary_line(report))
        typer.echo(f"  -> {path}")


@baseline_app.command("bless")
def bless(
    source: Path = typer.Option(Path("results"), "--from", help="Directory holding runs/*.json."),
    reason: str = typer.Option(
        ..., help="Why this is the new baseline. Required, may not be blank."
    ),
) -> None:
    """Freeze the current runs as the baseline the gate compares against.

    A re-bless is its own commit, touching nothing else, with a message that
    says why (how-to-run.md) -- enforced here by refusing a blank reason.
    """
    runs_dir = source / "runs"
    if not runs_dir.is_dir():
        typer.echo(f"no runs found at {runs_dir}; run `gt run --all` first", err=True)
        raise typer.Exit(code=1)

    reports: dict[str, ScoringReport] = {}
    for path in sorted(runs_dir.glob("*.json")):
        try:
            report = read_report(path)
        except ReportIOError as exc:
            typer.echo(f"FAIL: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        reports[report.config_name] = report

    if not reports:
        typer.echo(f"{runs_dir} has no result files to bless", err=True)
        raise typer.Exit(code=1)

    try:
        path = write_baseline(reports, source, reason=reason)
    except ReportIOError as exc:
        typer.echo(f"FAIL: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"blessed {len(reports)} configs -> {path}")
    for name in sorted(reports):
        typer.echo(f"  {_summary_line(reports[name])}")
