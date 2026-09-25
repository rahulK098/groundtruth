"""Value objects the scorer produces.

Everything here is a plain, immutable dataclass rather than a pydantic model.
These values never cross a validated boundary the way config or golden-set
records do -- they are produced and consumed entirely within this process,
serialized to JSON only by ``groundtruth.scoring.report_io`` -- so the extra
weight of pydantic buys nothing here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

#: Reported cutoffs, per methodology.md.
K_VALUES: Final[tuple[int, ...]] = (1, 3, 5, 10)

#: Recall, MRR, nDCG -- in this fixed order everywhere a metric name is used,
#: so report output and comparisons are never accidentally reordered.
METRIC_NAMES: Final[tuple[str, ...]] = ("recall", "mrr", "ndcg")


@dataclass(frozen=True, slots=True)
class QueryScore:
    """One query's metrics, or a record of why it was skipped.

    ``skipped`` is expected to always be ``False`` in practice --
    ``GoldenPair`` already forbids a query with no gain>=2 span -- but the
    field exists because the methodology asserts the exclusion rule anyway
    rather than trusting an upstream guarantee silently.
    """

    query_id: str
    category: str
    origin: str
    skipped: bool
    #: metric name -> {k -> value}. Empty for a skipped query.
    values: dict[str, dict[int, float]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """A macro-averaged metric, with a bootstrap 95% CI and its sample size.

    ``n`` matters as much as ``mean``: a category with n=3 and one with n=40
    print the same mean with wildly different confidence, and burying that
    in a single number is how a report overclaims.
    """

    mean: float
    ci_low: float
    ci_high: float
    n: int


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Wall-clock cost, reported because the recommendation argues a tradeoff."""

    embed_ms: float
    dense_ms: float
    lexical_ms: float
    fuse_ms: float
    rerank_ms: float
    total_ms: float


#: metric -> {k -> MetricSummary}
MetricTable = dict[str, dict[int, MetricSummary]]


@dataclass(frozen=True, slots=True)
class ScoringReport:
    """One configuration's complete, self-describing evaluation result.

    Carries all four identity components the gate compares against a
    baseline (ADR-0010), so a result file is never ambiguous about the
    exact conditions that produced it.
    """

    config_name: str
    config_hash: str
    golden_set_hash: str
    corpus_manifest_hash: str
    code_version: str
    run_fingerprint: str

    n_queries: int
    n_skipped: int

    overall: MetricTable
    by_category: dict[str, MetricTable]
    by_origin: dict[str, MetricTable]

    latency: LatencySummary
