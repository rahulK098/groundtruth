"""Locating the project's committed data directories.

Resolved by walking up for a ``pyproject.toml`` rather than counting parent
hops, because counting breaks the moment the package is installed non-editable
or the layout is rearranged.
"""

from __future__ import annotations

from pathlib import Path


class ProjectLayoutError(Exception):
    """The project root or one of its data directories could not be located."""


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ProjectLayoutError(
        "could not locate the project root: no ancestor of "
        f"{Path(__file__).resolve()} contains a pyproject.toml"
    )


def configs_dir() -> Path:
    return project_root() / "configs"


def tokenizers_dir() -> Path:
    return project_root() / "data" / "tokenizers"


def corpus_dir() -> Path:
    return project_root() / "data" / "corpus"


def cache_dir() -> Path:
    return project_root() / "data" / "cache"


def golden_dir() -> Path:
    return project_root() / "data" / "golden"
