"""``gt corpus`` commands."""

from __future__ import annotations

from pathlib import Path

import typer

from groundtruth.corpus.courtlistener import fetch_documents
from groundtruth.corpus.selection import CorpusSelection
from groundtruth.corpus.snapshot import (
    SnapshotError,
    read_manifest,
    verify_snapshot,
    write_snapshot,
)
from groundtruth.paths import corpus_dir
from groundtruth.settings import MissingCredentialError, Settings

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _progress(done: int, total: int) -> None:
    typer.echo(f"  ... {done}/{total}", err=True)


@app.command("fetch")
def fetch(
    limit: int = typer.Option(150, help="Number of opinions to keep."),
    courts: str = typer.Option("scotus", help="Comma-separated CourtListener court ids."),
    after: str = typer.Option("1990-01-01", help="Earliest date_filed (inclusive)."),
    before: str = typer.Option("2020-12-31", help="Latest date_filed (inclusive)."),
    max_chars: int = typer.Option(200_000, help="Skip opinions longer than this."),
    out: Path | None = typer.Option(None, help="Snapshot directory. Defaults to data/corpus."),
) -> None:
    """Fetch a deterministic slice of opinions and write a committed snapshot.

    Requires COURTLISTENER_API_TOKEN. This is the ONLY step that needs a
    credential or a network -- reproducing the comparison table needs neither,
    because the snapshot this writes is committed.
    """
    destination = out or corpus_dir()

    selection = CorpusSelection(
        courts=tuple(c.strip() for c in courts.split(",") if c.strip()),
        date_filed_after=after,
        date_filed_before=before,
        limit=limit,
        max_chars=max_chars,
    )

    try:
        token = Settings().require_courtlistener_token()
    except MissingCredentialError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"selection: {selection.corpus_id}")
    documents = fetch_documents(selection, token, on_progress=_progress)

    if not documents:
        typer.echo("No documents matched the selection.", err=True)
        raise typer.Exit(code=1)

    manifest = write_snapshot(
        documents,
        destination,
        corpus_id=selection.corpus_id,
        source="courtlistener-v4",
    )

    total_chars = sum(len(doc.text) for doc in documents)
    typer.echo(
        f"\nwrote {manifest.doc_count} documents to {destination}\n"
        f"  characters   : {total_chars:,}\n"
        f"  content hash : {manifest.content_sha256[:16]}...\n"
        f"  normalizer   : v{manifest.normalizer_version}\n"
        f"  manifest     : {manifest.manifest_hash[:23]}..."
    )


@app.command("verify")
def verify(
    directory: Path | None = typer.Option(
        None, help="Snapshot directory. Defaults to data/corpus."
    ),
) -> None:
    """Check the snapshot matches its manifest.

    Catches a corpus swapped underneath a committed baseline, and a snapshot
    written under different normalization rules -- the case where every
    golden-set character offset is silently wrong.
    """
    target = directory or corpus_dir()
    try:
        documents = verify_snapshot(target)
        manifest = read_manifest(target)
    except SnapshotError as exc:
        typer.echo(f"FAIL: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"OK: {len(documents)} documents, normalizer v{manifest.normalizer_version}, "
        f"content {manifest.content_sha256[:16]}..."
    )
