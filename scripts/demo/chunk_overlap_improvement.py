#!/usr/bin/env python
"""Demo 3: a legitimate improvement passes, on a document subset.

The brief's original third scenario is "reranker on, passes as an
improvement." The reranker is Phase 9 work -- Retriever itself refuses a
config with it enabled (see retrieval/pipeline.py) -- so there is nothing to
demo yet. Substituted with a change that is real today and serves the same
purpose: doubling dense_512's chunk_overlap (50 -> 100 tokens) gives every
chunk more surrounding context, which can only help or be neutral for
finding the right passage, never hurt it structurally.

Like the chunk-size regression demo, wider overlap produces novel chunk
windows over the whole corpus -- bounded here to the same small, documented
subset of real documents for the same runtime reason. Compared before-vs-after
on that subset, not against results/baseline.json.

Usage: uv run python scripts/demo/chunk_overlap_improvement.py
"""

from __future__ import annotations

import sys

from _common import gate_before_vs_after, report_result, score_before_and_after, summarize

from groundtruth.config.registry import load_config, shipped_configs_dir

DEMO_NAME = "chunk_overlap_improvement"
CONFIG_NAME = "dense_512"

NEW_CHUNK_OVERLAP = 100


def main() -> int:
    config = load_config(shipped_configs_dir() / f"{CONFIG_NAME}.yaml")
    improved = config.model_copy(
        update={"chunking": config.chunking.model_copy(update={"chunk_overlap": NEW_CHUNK_OVERLAP})}
    )

    before, after, n_queries = score_before_and_after(config, improved)
    result = gate_before_vs_after(before, after, CONFIG_NAME)

    summary = (
        f"scope: {n_queries} golden queries, a documented subset of documents "
        f"(see _common.py:SUBSET_SIZE) -- NOT the full corpus\n\n"
        f"before (chunk_overlap=50):\n{summarize(before)}\n\n"
        f"after  (chunk_overlap={NEW_CHUNK_OVERLAP}):\n{summarize(after)}"
    )
    return report_result(DEMO_NAME, summary, result, expect_pass=True)


if __name__ == "__main__":
    sys.exit(main())
