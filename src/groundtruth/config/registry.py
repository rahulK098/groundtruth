"""Loading retrieval configurations from YAML.

Turns human-authored files into validated, hashed configurations. Its most
important job is refusing two configurations that are materially identical
under different names: running both wastes time and, worse, reports a single
experiment as two independent data points in the comparison table.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import yaml
from pydantic import ValidationError

from groundtruth.config.models import RetrievalConfig

CONFIG_SUFFIXES: Final[frozenset[str]] = frozenset({".yaml", ".yml"})

#: Files that live alongside the configs but are a different kind of document.
#: Without this, loading the directory would try to validate the gate policy as
#: a retrieval config and fail confusingly.
NON_CONFIG_STEMS: Final[frozenset[str]] = frozenset({"gate_policy"})


class ConfigError(Exception):
    """A configuration could not be loaded, parsed, or validated."""


def shipped_configs_dir() -> Path:
    """Locate the project's ``configs/`` directory.

    Walks up from this module looking for a directory that has both a
    ``configs/`` child and a ``pyproject.toml``, rather than counting parent
    hops -- counting breaks the moment the package is installed non-editable
    or the layout is rearranged.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "configs"
        if candidate.is_dir() and (parent / "pyproject.toml").is_file():
            return candidate
    raise ConfigError(
        "could not locate the shipped configs directory; expected a project "
        "root containing both 'configs/' and 'pyproject.toml'"
    )


def load_config(path: Path) -> RetrievalConfig:
    """Load and validate a single configuration file."""
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path.name}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"could not read {path.name}: {exc}") from exc

    if not isinstance(raw, Mapping):
        kind = type(raw).__name__
        raise ConfigError(f"{path.name} must contain a YAML mapping, got {kind}")

    try:
        config = RetrievalConfig.model_validate(dict(raw))
    except ValidationError as exc:
        # Name the file. A validation failure while loading a whole directory
        # is close to useless if it does not say which file is wrong.
        raise ConfigError(f"{path.name} is not a valid configuration: {exc}") from exc

    if config.name != path.stem:
        raise ConfigError(
            f"{path.name}: filename must match the declared name {config.name!r}; "
            f"results are written per config name, so a mismatch makes the file "
            f"on disk and the row in the report disagree about what ran"
        )

    return config


def load_all_configs(directory: Path) -> Mapping[str, RetrievalConfig]:
    """Load every configuration in a directory, keyed by name."""
    if not directory.is_dir():
        raise ConfigError(f"configuration directory not found: {directory}")

    paths = sorted(
        path
        for path in directory.iterdir()
        if path.suffix in CONFIG_SUFFIXES and path.stem not in NON_CONFIG_STEMS
    )
    if not paths:
        # Returning an empty mapping would make the entire evaluation a no-op
        # that reports success.
        raise ConfigError(f"no configuration files found in {directory}")

    configs: dict[str, RetrievalConfig] = {}
    names_by_hash: dict[str, str] = {}

    for path in paths:
        config = load_config(path)

        if config.name in configs:
            raise ConfigError(f"duplicate configuration name {config.name!r} in {directory}")

        previous = names_by_hash.get(config.config_hash)
        if previous is not None:
            raise ConfigError(
                f"configurations {previous!r} and {config.name!r} are identical "
                f"(both hash to {config.config_hash}); comparing them would "
                f"report one experiment as two independent results"
            )

        names_by_hash[config.config_hash] = config.name
        configs[config.name] = config

    return configs
