"""Storage primitives: the only layer that differs between the two paths.

Everything above this package -- fusion, ordering, truncation, reranking -- is
one shared code path (ADR-0001). What varies is where candidates come from:
NumPy and in-process BM25 for the evaluation path the gate measures, Postgres
for the service path.

Deliberately imports nothing from its own submodules, so ``retrieval`` can
depend on :mod:`groundtruth.index.models` without dragging in NumPy indexes.
"""
