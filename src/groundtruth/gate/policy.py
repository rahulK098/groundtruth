"""Loading the gate policy: thresholds and gated metrics live in YAML, never
hardcoded in the test (ADR-0006).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError

from groundtruth.frozen import FrozenModel


class GatePolicyError(Exception):
    """The gate policy could not be loaded, parsed, or validated."""


class GatePolicy(FrozenModel):
    """The committed thresholds a run is judged against.

    Read once, at collection time, by ``tests/gate/test_regression_gate.py``
    -- chosen before any number exists to be tempted by, per the policy
    file's own header comment.
    """

    primary_metric: str = Field(min_length=1)
    gated_configs: tuple[str, ...] = Field(min_length=1)
    #: "{metric}_at_{k}" -> the absolute drop, in metric points, tolerated
    #: before the gate fails that (config, metric, k) triple.
    max_absolute_drop: dict[str, float]

    #: The anti-cheat asymmetry (ADR-0010): a golden-set or corpus change is
    #: a hard fail by default, because either one moving the goalposts is
    #: indistinguishable from a regression "fixed" by deleting hard queries.
    require_golden_set_hash_match: bool = True
    require_corpus_hash_match: bool = True
    #: A config change IS the workflow, so this defaults to a warning rather
    #: than a failure -- failing here would make the gate useless within a day.
    config_hash_mismatch: Literal["warn", "fail"] = "warn"
    #: Per-category metrics are for diagnosis only; at n~20 a single query
    #: swings a category by several points, and gating on them would make a
    #: flapping gate people learn to ignore.
    per_category: Literal["report_only"] = "report_only"


def load_gate_policy(path: Path) -> GatePolicy:
    if not path.is_file():
        raise GatePolicyError(f"gate policy not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise GatePolicyError(f"could not parse {path.name}: {exc}") from exc

    try:
        return GatePolicy.model_validate(raw)
    except ValidationError as exc:
        raise GatePolicyError(f"{path.name} is not a valid gate policy: {exc}") from exc
