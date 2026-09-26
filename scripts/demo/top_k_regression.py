#!/usr/bin/env python
"""Demo 1: drop dense_512's top_k from 10 to 2. Expect recall_at_10 to fail.

The most direct way to break retrieval quality: return fewer results than
the golden set is scored at. Needs no new embeddings -- dense_512's 512-token
chunks are already in the committed cache -- so this demo runs in
milliseconds.

Usage: uv run python scripts/demo/top_k_regression.py
"""

from __future__ import annotations

import sys

from _common import gate_against_baseline, report_result, score_modified_config, summarize

from groundtruth.config.registry import load_config, shipped_configs_dir

DEMO_NAME = "top_k_regression"
CONFIG_NAME = "dense_512"


def main() -> int:
    config = load_config(shipped_configs_dir() / f"{CONFIG_NAME}.yaml")
    broken = config.model_copy(update={"top_k": 2})

    report = score_modified_config(broken)
    result = gate_against_baseline(report, CONFIG_NAME)

    return report_result(DEMO_NAME, summarize(report), result, expect_pass=False)


if __name__ == "__main__":
    sys.exit(main())
