"""Macro-averaging over queries, with a bootstrap confidence interval.

Macro, not pooled: categories are unbalanced (methodology.md), so a per-query
mean-of-means is what keeps a large category from swamping a small one when
the two are compared side by side.

The bootstrap is the only way this project can attach an interval to a mean
of ~20-90 numbers without an assumption about their distribution -- and
without it, "0.82 vs 0.79" reads as a real difference when it may be noise.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from itertools import groupby
from typing import Final

from groundtruth.scoring.models import MetricSummary, MetricTable, QueryScore

#: Per methodology.md: 1000 resamples, seed 20240101, published everywhere a
#: mean is. The seed is fixed so two runs of the same data report the same
#: interval -- a CI that moved between two identical runs would itself look
#: like a regression.
DEFAULT_RESAMPLES: Final[int] = 1000
DEFAULT_SEED: Final[int] = 20240101

#: A 95% interval: 2.5th and 97.5th percentiles of the resampled means.
_LOWER_PERCENTILE: Final[float] = 0.025
_UPPER_PERCENTILE: Final[float] = 0.975


def bootstrap_ci(
    values: Sequence[float], *, resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED
) -> MetricSummary:
    """Mean and a percentile bootstrap 95% CI over ``values``.

    Degenerates gracefully at the edges: zero values reports a 0.0 mean with
    n=0 rather than raising, and a single value (or constant data) reports a
    CI equal to the mean itself rather than a fabricated spread.
    """
    n = len(values)
    if n == 0:
        return MetricSummary(mean=0.0, ci_low=0.0, ci_high=0.0, n=0)

    mean = statistics.fmean(values)
    if n == 1:
        return MetricSummary(mean=mean, ci_low=mean, ci_high=mean, n=1)

    rng = random.Random(seed)
    resample_means = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(resamples)
    )

    low_index = max(0, int(_LOWER_PERCENTILE * resamples))
    high_index = min(resamples - 1, int(_UPPER_PERCENTILE * resamples))
    return MetricSummary(
        mean=mean, ci_low=resample_means[low_index], ci_high=resample_means[high_index], n=n
    )


def macro_average(
    scores: Sequence[QueryScore], *, resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED
) -> MetricTable:
    """Aggregate a batch of per-query scores into one metric table.

    Skipped queries contribute nothing -- their ``values`` are empty by
    construction, so they are naturally absent from every (metric, k) sample
    without special-casing here.
    """
    samples: dict[str, dict[int, list[float]]] = {}
    for query in scores:
        for metric, by_k in query.values.items():
            for k, value in by_k.items():
                samples.setdefault(metric, {}).setdefault(k, []).append(value)

    table: MetricTable = {}
    for metric, values_by_k in samples.items():
        table[metric] = {
            k: bootstrap_ci(values, resamples=resamples, seed=seed)
            for k, values in values_by_k.items()
        }
    return table


def group_by(scores: Sequence[QueryScore], key: str) -> dict[str, MetricTable]:
    """Macro-average within each distinct value of ``category`` or ``origin``.

    A plain groupby rather than a defaultdict-of-lists: scores are already
    produced in a stable order, and sorting first keeps this a pure function
    of its input regardless of what order the caller happened to build it in.
    """
    getter = (lambda q: q.category) if key == "category" else (lambda q: q.origin)
    ordered = sorted(scores, key=getter)
    return {group: macro_average(list(items)) for group, items in groupby(ordered, key=getter)}
