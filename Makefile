# Groundtruth. See docs/how-to-run.md for the full explanation of every
# target here -- this file is intentionally thin, one line per command,
# so it never drifts from what that document promises.

.PHONY: reproduce verify gate demo-top-k demo-chunk-size demo-improvement \
        demo-regression demos

# "The one command" (docs/how-to-run.md). gt report (the markdown
# comparison table) is Phase 12 work and not implemented yet -- everything
# up to the frozen numbers in results/baseline.json runs today.
reproduce:
	uv sync --frozen --extra dev
	uv run pytest
	@echo "gt report is not implemented yet (Phase 12); see results/baseline.json for the frozen numbers"

# Corpus and cache integrity, independent of the gate's own comparison.
verify:
	uv run gt corpus verify
	uv run gt cache verify

# The regression gate itself (ADR-0006): local, no CI, no minutes, no secrets.
gate: verify
	uv run pytest -m gate

# Three self-reverting scenarios (docs/how-to-run.md). None writes to a
# tracked file -- see scripts/demo/_common.py for why "revert" here means
# "never touch configs/ or data/cache/ in the first place."
demo-top-k:
	uv run python scripts/demo/top_k_regression.py

demo-chunk-size:
	uv run python scripts/demo/chunk_size_regression.py

demo-improvement:
	uv run python scripts/demo/chunk_overlap_improvement.py

# The two regressions together -- what "make demo-regression" in
# how-to-run.md refers to.
demo-regression: demo-top-k demo-chunk-size

# All three, for a single "prove every claim" pass.
demos: demo-regression demo-improvement
