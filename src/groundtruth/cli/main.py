"""The ``gt`` command line interface.

Thin wrappers only. Every command delegates to a library function so the same
code path is exercised by tests, by the harness, and by a human at a terminal.
"""

from __future__ import annotations

import typer

from groundtruth.cli import corpus as corpus_commands

app = typer.Typer(
    name="gt",
    help="Groundtruth: a regression-gated retrieval evaluation harness.",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(corpus_commands.app, name="corpus", help="Fetch and verify the corpus snapshot.")


@app.command()
def version() -> None:
    """Print the harness version."""
    from groundtruth import __version__

    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
