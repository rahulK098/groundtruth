#!/usr/bin/env python
"""Demo 4: turn the reranker on. Expected by the brief to improve. It does not.

Before: hybrid_512. After: hybrid_512_rerank -- identical except the
cross-encoder stage is enabled. Both are scored on the FULL corpus and the
full golden set, served entirely from committed caches (ADR-0003, ADR-0012):
no model, no network, a few seconds.

The brief's third scenario assumed "rerank on" would pass as an improvement.
Measured, it does the opposite: recall@10 inches up (0.861 -> 0.867) but
mrr@10 (0.624 -> 0.580) and ndcg@10 (0.607 -> 0.578) both fall past the
gate's tolerance, at a recorded ~105 s per query on single-threaded CPU. This
is the scenario the harness exists for: a change everyone expects to help,
caught by a number before it ships. The demo asserts what was measured, not
what was assumed. The "gate is not always red" proof lives in
chunk_overlap_improvement.py.

Compared before-vs-after through the same evaluate_config the gate uses. The
two configs are different identities (different config_hash), so the gate
itself never compares them to each other -- it compares each against its own
baseline. This demo answers the question a reader actually has: "what did
turning the reranker on do?"

Usage: uv run python scripts/demo/rerank_on.py
"""

from __future__ import annotations

import sys

from _common import gate_before_vs_after, report_result, summarize

from groundtruth import __version__
from groundtruth.config.registry import load_config, shipped_configs_dir
from groundtruth.corpus.snapshot import read_manifest, verify_snapshot
from groundtruth.golden.store import read_golden_set
from groundtruth.paths import cache_dir, corpus_dir, golden_dir
from groundtruth.retrieval.build import build_cached_retriever
from groundtruth.scoring.evaluate import score_config
from groundtruth.scoring.models import ScoringReport

DEMO_NAME = "rerank_on"
BEFORE = "hybrid_512"
AFTER = "hybrid_512_rerank"


def _score(name: str) -> ScoringReport:
    config = load_config(shipped_configs_dir() / f"{name}.yaml")
    documents = verify_snapshot(corpus_dir())
    return score_config(
        build_cached_retriever(config, documents, cache_dir()),
        read_golden_set(golden_dir()),
        corpus_manifest_hash=read_manifest(corpus_dir()).manifest_hash,
        code_version=__version__,
    )


def main() -> int:
    before, after = _score(BEFORE), _score(AFTER)
    result = gate_before_vs_after(before, after, AFTER)

    summary = (
        f"scope: full corpus, {before.n_queries} golden queries\n\n"
        f"before ({BEFORE}):\n{summarize(before)}\n"
        f"latency p50   : {before.latency.total_ms:.1f} ms\n\n"
        f"after  ({AFTER}):\n{summarize(after)}\n"
        f"latency p50   : {after.latency.total_ms:.1f} ms "
        f"(rerank stage {after.latency.rerank_ms:.1f} ms, recorded at cache build)"
    )
    return report_result(DEMO_NAME, summary, result, expect_pass=False)


if __name__ == "__main__":
    sys.exit(main())
