"""``gt golden`` commands.

The interaction loop for the review, and nothing else: which candidate comes
next, how it is rendered, and what a decision becomes all live in
:mod:`groundtruth.golden.review`, where they are tested without a terminal.

Every decision is appended to the log **before** the next candidate is shown.
The review is the project's critical path (ADR-0009), and a crash or a `q`
must never cost more than the candidate on screen.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import typer

from groundtruth.corpus.models import Document
from groundtruth.corpus.snapshot import SnapshotError, verify_snapshot
from groundtruth.golden.authoring import (
    AuthoringError,
    from_authored,
    next_human_query_id,
    to_template,
)
from groundtruth.golden.candidates import Candidate, ReviewDecision
from groundtruth.golden.generate import generate_candidates
from groundtruth.golden.review import (
    ReviewContext,
    ReviewError,
    accept,
    from_editable,
    pending,
    reject,
    render_candidate,
    skip,
    source_passage,
    to_editable,
)
from groundtruth.golden.sampling import (
    DEFAULT_SEED,
    DEFAULT_WINDOW_CHARS,
    SampledPassage,
    sample_passages,
)
from groundtruth.golden.store import (
    CANDIDATES_FILENAME,
    GoldenStoreError,
    append_candidates,
    append_decision,
    read_candidates,
    read_golden_set,
    read_review_log,
    rebuild_golden_set,
    review_summary,
)
from groundtruth.golden.validation import errors, validate
from groundtruth.llm.models import ProviderNotConfiguredError
from groundtruth.llm.registry import PROVIDER_NAMES, build_provider
from groundtruth.paths import corpus_dir, golden_dir
from groundtruth.settings import Settings

app = typer.Typer(no_args_is_help=True, add_completion=False)

#: Indirection so tests can substitute the editor. ``typer.edit`` returns
#: ``None`` when the editor closes without a change.
_edit = typer.edit

_DIRECTORY_HELP = "Golden-set directory. Defaults to data/golden."
_CORPUS_HELP = "Corpus snapshot directory. Defaults to data/corpus."
_REVIEWER_HELP = "Recorded on every decision. Also read from GT_REVIEWER."

_PROMPT = "[a]ccept  [e]dit  [r]eject  [s]kip  [q]uit"
_CHOICES = ("a", "e", "r", "s", "q")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _progress(done: int, total: int) -> None:
    typer.echo(f"  ... {done}/{total}", err=True)


def _load_corpus(directory: Path | None) -> dict[str, Document]:
    try:
        documents = verify_snapshot(directory or corpus_dir())
    except SnapshotError as exc:
        typer.echo(f"FAIL: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    return {doc.doc_id: doc for doc in documents}


def _fail(message: str) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(code=1)


# --- generate ---------------------------------------------------------------


@app.command("generate")
def generate(
    count: int = typer.Option(120, help="Passages to sample, one proposal each."),
    provider: str = typer.Option(
        "anthropic",
        envvar="GT_LLM_PROVIDER",
        help=f"One of: {', '.join(PROVIDER_NAMES)}.",
    ),
    model: str = typer.Option(
        "", help="Model id, or the deployment name on Azure. Defaults per provider."
    ),
    window: int = typer.Option(DEFAULT_WINDOW_CHARS, help="Passage window in characters."),
    seed: int = typer.Option(DEFAULT_SEED, help="Sampling seed."),
    directory: Path | None = typer.Option(None, help=_DIRECTORY_HELP),
    corpus: Path | None = typer.Option(None, help=_CORPUS_HELP),
) -> None:
    """Propose candidates from sampled passages. Proposals only, never labels.

    Needs credentials for the chosen provider and nothing else -- no extra to
    install. This is the only golden command that calls a model; review,
    rebuild and validate never do.

    There is deliberately no fallback to another provider: a run answered by
    whichever vendor happened to be reachable would not be reproducible.
    """
    target = directory or golden_dir()
    if (target / CANDIDATES_FILENAME).exists():
        # Candidate ids are positional (c-0001...), so a second run would
        # collide with the first and the review log would point at the wrong
        # proposals. Regenerating means starting the review over.
        raise _fail(
            f"{target / CANDIDATES_FILENAME} already exists. Delete it (and the "
            f"review log, which refers to it) to regenerate."
        )

    try:
        chat = build_provider(provider, Settings(), model=model or None)
    except ProviderNotConfiguredError as exc:
        raise _fail(str(exc)) from exc

    documents = _load_corpus(corpus)
    passages = sample_passages(
        tuple(documents.values()), count=count, window_chars=window, seed=seed
    )
    typer.echo(
        f"sampled {len(passages)} passages from {len(documents)} documents\n"
        f"provider: {chat.name}  model: {chat.model}"
    )

    candidates, report = generate_candidates(passages, chat, on_progress=_progress)

    append_candidates(candidates, target)

    lines = [
        f"\nwrote {report.accepted} candidates to {target / CANDIDATES_FILENAME}",
        f"  provider : {report.provider}",
        f"  model    : {report.model}",
        f"  prompt   : {report.prompt_version}",
        f"  rejected : {len(report.rejected)}",
    ]
    for code, n in sorted(report.rejection_counts.items()):
        lines.append(f"    {code:<18} {n}")
    typer.echo("\n".join(lines))


# --- add --------------------------------------------------------------------


@app.command("add")
def add(
    reviewer: str = typer.Option(..., envvar="GT_REVIEWER", help=_REVIEWER_HELP),
    doc_id: str = typer.Option("", help="Pre-fill the first span's document id."),
    directory: Path | None = typer.Option(None, help=_DIRECTORY_HELP),
    corpus: Path | None = typer.Option(None, help=_CORPUS_HELP),
) -> None:
    """Write one pair by hand, in an editor. Recorded as human-authored."""
    target = directory or golden_dir()
    documents = _load_corpus(corpus)

    text = _edit(to_template(doc_id), extension=".yaml")
    if text is None:
        typer.echo("no change; nothing added")
        return

    context = ReviewContext(reviewer=reviewer, decided_at=_now())
    query_id = next_human_query_id(read_review_log(target))
    try:
        decision = from_authored(text, documents, query_id=query_id, context=context)
    except AuthoringError as exc:
        raise _fail(f"not added: {exc}") from exc

    append_decision(decision, target)
    golden = rebuild_golden_set(target)
    typer.echo(f"added {query_id}; golden set now has {len(golden.pairs)} pairs")


# --- review -----------------------------------------------------------------


def _edit_candidate(candidate: Candidate, passage: SampledPassage) -> Candidate | None:
    """Open the candidate in an editor; None if unchanged or the edit is bad."""
    text = _edit(to_editable(candidate), extension=".yaml")
    if text is None:
        typer.echo("no change")
        return None
    try:
        return from_editable(text, candidate, passage)
    except ReviewError as exc:
        typer.echo(f"edit refused: {exc}", err=True)
        return None


class _Quit:
    """The reviewer pressed `q`. A distinct type so the caller can narrow on it."""


def _decide(
    candidate: Candidate,
    passage: SampledPassage,
    context: ReviewContext,
    started: float,
) -> ReviewDecision | _Quit | None:
    """One keystroke. A decision, None to re-prompt, or a quit."""
    choice = typer.prompt(_PROMPT, type=str).strip().lower()[:1]
    if choice not in _CHOICES:
        return None

    seconds = round(time.monotonic() - started, 1)
    if choice == "q":
        return _Quit()
    if choice == "a":
        return accept(candidate, context, seconds=seconds)
    if choice == "s":
        return skip(candidate, context, seconds=seconds)
    if choice == "r":
        reason = typer.prompt("reason", default="", show_default=False).strip()
        if not reason:
            typer.echo("a rejection needs a reason")
            return None
        return reject(candidate, context, reason, seconds=seconds)

    revised = _edit_candidate(candidate, passage)
    if revised is None:
        return None
    return accept(candidate, context, revised=revised, seconds=seconds)


@app.command("review")
def review(
    reviewer: str = typer.Option(..., envvar="GT_REVIEWER", help=_REVIEWER_HELP),
    directory: Path | None = typer.Option(None, help=_DIRECTORY_HELP),
    corpus: Path | None = typer.Option(None, help=_CORPUS_HELP),
) -> None:
    """Review candidates one at a time. Every decision is logged immediately.

    Quit with `q` at any point; skipped candidates come back next time.
    """
    target = directory or golden_dir()
    documents = _load_corpus(corpus)

    try:
        queue: Sequence[Candidate] = pending(read_candidates(target), read_review_log(target))
    except GoldenStoreError as exc:
        raise _fail(str(exc)) from exc

    if not queue:
        typer.echo("nothing to review")
        return

    total = len(queue)
    for position, candidate in enumerate(queue, start=1):
        document = documents.get(candidate.source_doc_id)
        if document is None:
            typer.echo(
                f"{candidate.candidate_id}: source {candidate.source_doc_id!r} is not in the "
                f"corpus; skipping",
                err=True,
            )
            continue
        passage = source_passage(candidate, document.text)

        typer.echo("\n" + render_candidate(candidate, passage, position, total) + "\n")
        started = time.monotonic()
        outcome: ReviewDecision | _Quit | None = None
        while outcome is None:
            context = ReviewContext(reviewer=reviewer, decided_at=_now())
            outcome = _decide(candidate, passage, context, started)

        if isinstance(outcome, _Quit):
            break
        append_decision(outcome, target)

    golden = rebuild_golden_set(target)
    summary = review_summary(target)
    typer.echo(
        f"\ngolden set: {len(golden.pairs)} pairs  "
        f"(accept {summary['accept']}, edit {summary['edit']}, "
        f"reject {summary['reject']}, skip {summary['skip']})"
    )


# --- rebuild / validate / status --------------------------------------------


@app.command("rebuild")
def rebuild(directory: Path | None = typer.Option(None, help=_DIRECTORY_HELP)) -> None:
    """Rewrite golden_set.jsonl from the review log. Safe to run any time."""
    target = directory or golden_dir()
    try:
        golden = rebuild_golden_set(target)
    except GoldenStoreError as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"wrote {len(golden.pairs)} pairs  {golden.golden_set_hash}")


@app.command("validate")
def validate_command(
    directory: Path | None = typer.Option(None, help=_DIRECTORY_HELP),
    corpus: Path | None = typer.Option(None, help=_CORPUS_HELP),
) -> None:
    """Check every label against the corpus, and the set against its targets.

    Errors (a label the corpus no longer matches) exit non-zero. Warnings (the
    set is not finished) are printed and do not.
    """
    target = directory or golden_dir()
    documents = _load_corpus(corpus)
    try:
        golden = read_golden_set(target)
    except GoldenStoreError as exc:
        raise _fail(str(exc)) from exc

    issues = validate(golden, tuple(documents.values()))
    for issue in issues:
        typer.echo(str(issue), err=issue.severity == "error")

    failed = errors(issues)
    typer.echo(
        f"{len(golden.pairs)} pairs, {len(failed)} errors, {len(issues) - len(failed)} warnings"
    )
    if failed:
        raise typer.Exit(code=1)


@app.command("status")
def status(directory: Path | None = typer.Option(None, help=_DIRECTORY_HELP)) -> None:
    """The counts the methodology quotes: generated, accepted, edited, rejected."""
    target = directory or golden_dir()
    try:
        summary = review_summary(target)
        golden = read_golden_set(target)
    except GoldenStoreError as exc:
        raise _fail(str(exc)) from exc

    lines = [
        f"candidates generated : {summary['candidates_generated']}",
        f"decisions recorded   : {summary['decisions_recorded']} "
        f"({summary['superseded']} superseded)",
        f"  accept : {summary['accept']}",
        f"  edit   : {summary['edit']}",
        f"  reject : {summary['reject']}",
        f"  skip   : {summary['skip']}",
        "",
        f"golden set : {len(golden.pairs)} pairs  {golden.golden_set_hash}",
    ]
    for origin, n in golden.counts_by_origin().items():
        lines.append(f"  {origin:<22} {n}")
    for category, n in golden.counts_by_category().items():
        lines.append(f"  {category:<22} {n}")
    typer.echo("\n".join(lines))
