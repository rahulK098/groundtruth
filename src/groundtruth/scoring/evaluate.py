"""Scoring one configuration's retriever against the golden set.

The one function every entry point (`gt run`, the regression gate) calls.
Everything downstream of it -- span resolution, the three metrics,
macro-averaging -- is pure and tested in isolation; this module's own job is
just wiring: run every query, collect what each metric function reports, and
assemble the identity-carrying report the gate compares.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Final

from groundtruth.config.hashing import run_fingerprint
from groundtruth.golden.models import GoldenPair, GoldenSet
from groundtruth.retrieval.models import StageLatenciesMs
from groundtruth.retrieval.pipeline import Retriever
from groundtruth.scoring.aggregate import group_by, macro_average
from groundtruth.scoring.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from groundtruth.scoring.models import K_VALUES, LatencySummary, QueryScore, ScoringReport

#: name -> function, in the fixed order metric tables always report.
_METRIC_FUNCTIONS: Final = {
    "recall": recall_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}


def _score_one_query(
    retriever: Retriever, pair: GoldenPair, k_values: Sequence[int]
) -> tuple[QueryScore, StageLatenciesMs]:
    result = retriever.retrieve(pair.query, top_k=max(k_values))

    values: dict[str, dict[int, float]] = {}
    for metric_name, scorer in _METRIC_FUNCTIONS.items():
        by_k: dict[int, float] = {}
        for k in k_values:
            score = scorer(result.passages, pair.labels, k)
            if score is None:
                # Every metric function agrees on when a query is unscoreable
                # (no gain>=2 span), so the first None settles it for all
                # three -- there is nothing partial to record.
                return (
                    QueryScore(
                        query_id=pair.query_id,
                        category=pair.category,
                        origin=pair.provenance.origin,
                        skipped=True,
                        values={},
                    ),
                    result.latency_ms,
                )
            by_k[k] = score
        values[metric_name] = by_k

    return (
        QueryScore(
            query_id=pair.query_id,
            category=pair.category,
            origin=pair.provenance.origin,
            skipped=False,
            values=values,
        ),
        result.latency_ms,
    )


def _aggregate_latency(latencies: Sequence[StageLatenciesMs]) -> LatencySummary:
    if not latencies:
        return LatencySummary(
            embed_ms=0.0, dense_ms=0.0, lexical_ms=0.0, fuse_ms=0.0, rerank_ms=0.0, total_ms=0.0
        )
    return LatencySummary(
        embed_ms=statistics.median(latency.embed for latency in latencies),
        dense_ms=statistics.median(latency.dense for latency in latencies),
        lexical_ms=statistics.median(latency.lexical for latency in latencies),
        fuse_ms=statistics.median(latency.fuse for latency in latencies),
        rerank_ms=statistics.median(latency.rerank for latency in latencies),
        total_ms=statistics.median(latency.total for latency in latencies),
    )


def score_config(
    retriever: Retriever,
    golden: GoldenSet,
    *,
    corpus_manifest_hash: str,
    code_version: str,
    k_values: Sequence[int] = K_VALUES,
) -> ScoringReport:
    """Run every golden-set query through ``retriever`` and score it.

    ``k_values`` is passed explicitly to ``retriever.retrieve`` rather than
    relying on the config's own ``top_k`` -- the scorer's k-values are an
    evaluation-time decision (methodology.md: reported at k in {1,3,5,10})
    independent of what any one configuration happens to set.
    """
    scores: list[QueryScore] = []
    latencies: list[StageLatenciesMs] = []
    for pair in golden.pairs:
        score, latency = _score_one_query(retriever, pair, k_values)
        scores.append(score)
        latencies.append(latency)

    config = retriever.config
    fingerprint = run_fingerprint(
        config_hash=config.config_hash,
        golden_set_hash=golden.golden_set_hash,
        corpus_manifest_hash=corpus_manifest_hash,
        code_version=code_version,
    )

    return ScoringReport(
        config_name=config.name,
        config_hash=config.config_hash,
        golden_set_hash=golden.golden_set_hash,
        corpus_manifest_hash=corpus_manifest_hash,
        code_version=code_version,
        run_fingerprint=fingerprint,
        n_queries=len(golden.pairs),
        n_skipped=sum(1 for score in scores if score.skipped),
        overall=macro_average(scores),
        by_category=group_by(scores, "category"),
        by_origin=group_by(scores, "origin"),
        latency=_aggregate_latency(latencies),
    )
