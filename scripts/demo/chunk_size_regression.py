#!/usr/bin/env python
"""Demo 2: shrink dense_512's chunk_size to 64 tokens, on a document subset.

Expected to hurt ranking quality (ndcg) more than whether the right chunk
was found at all (recall) -- a 64-token window still often contains enough
of a holding to satisfy the overlap policy, but fragments it across more
candidates, muddying the ranking. This is the demo that proves the gate is
multi-metric: a config can regress on one axis without regressing on another.

64-token chunks are novel strings, never embedded before, over EVERY
document in the corpus -- tens of thousands of fresh CPU forward passes,
hours rather than a demo. Scoped instead to a small, documented subset of
real documents (see scripts/demo/_common.py:SUBSET_SIZE): same real model,
same real code, same real metrics, bounded runtime. Compared before-vs-after
on that subset, not against results/baseline.json -- see
score_before_and_after's docstring for why.

Usage: uv run python scripts/demo/chunk_size_regression.py
"""

from __future__ import annotations

import sys

from _common import gate_before_vs_after, report_result, score_before_and_after, summarize

from groundtruth.config.registry import load_config, shipped_configs_dir

DEMO_NAME = "chunk_size_regression"
CONFIG_NAME = "dense_512"

#: ~10% of the window, the same proportion dense_256 uses relative to 512.
NEW_CHUNK_SIZE = 64
NEW_CHUNK_OVERLAP = 6


def main() -> int:
    config = load_config(shipped_configs_dir() / f"{CONFIG_NAME}.yaml")
    broken = config.model_copy(
        update={
            "chunking": config.chunking.model_copy(
                update={"chunk_size": NEW_CHUNK_SIZE, "chunk_overlap": NEW_CHUNK_OVERLAP}
            )
        }
    )

    before, after, n_queries = score_before_and_after(config, broken)
    result = gate_before_vs_after(before, after, CONFIG_NAME)

    summary = (
        f"scope: {n_queries} golden queries, a documented subset of documents "
        f"(see _common.py:SUBSET_SIZE) -- NOT the full corpus\n\n"
        f"before (chunk_size=512):\n{summarize(before)}\n\n"
        f"after  (chunk_size={NEW_CHUNK_SIZE}):\n{summarize(after)}"
    )
    return report_result(DEMO_NAME, summary, result, expect_pass=False)


if __name__ == "__main__":
    sys.exit(main())
