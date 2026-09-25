"""The ``gt`` command line interface.

Thin wrappers only. Every command delegates to a library function so the same
code path is exercised by tests, by the harness, and by a human at a terminal.
"""

from __future__ import annotations

import typer

from groundtruth.cli import cache as cache_commands
from groundtruth.cli import corpus as corpus_commands
from groundtruth.cli import golden as golden_commands
from groundtruth.cli import run as run_commands
from groundtruth.cli import search as search_commands

app = typer.Typer(
    name="gt",
    help="Groundtruth: a regression-gated retrieval evaluation harness.",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(corpus_commands.app, name="corpus", help="Fetch and verify the corpus snapshot.")
app.add_typer(cache_commands.app, name="cache", help="Build and verify the embedding cache.")
app.add_typer(
    golden_commands.app, name="golden", help="Propose, review and validate the golden set."
)
# Single commands, not groups -- registered directly rather than wrapped in a
# sub-Typer, which would demand a subcommand after `gt search` / `gt run`.
app.command("search")(search_commands.search)
app.command("run")(run_commands.run)
app.add_typer(
    run_commands.baseline_app, name="baseline", help="Bless the current runs as the baseline."
)


@app.command()
def version() -> None:
    """Print the harness version."""
    from groundtruth import __version__

    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
