"""The regression gate: a pure function, (baseline, current, policy) -> report.

Nothing here reads a file, builds a retriever, or knows what a corpus is --
that wiring lives in `tests/gate/test_regression_gate.py`, the one place
that actually IS the gate ADR-0006 describes: a real pytest test, run
locally with `pytest -m gate` or `make gate`, never in CI.
"""
