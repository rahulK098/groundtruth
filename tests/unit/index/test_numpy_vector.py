"""Exact cosine search over a dense matrix.

This is the dense arm the gate measures (ADR-0001). It is exact, so the
assertions here are hand-checkable arithmetic rather than "roughly the right
neighbours".
"""

from __future__ import annotations

import numpy as np
import pytest

from groundtruth.index.numpy_vector import NumpyVectorIndex, VectorIndexError

ROOT_HALF = float(np.sqrt(0.5))


def build(rows: dict[str, list[float]]) -> NumpyVectorIndex:
    return NumpyVectorIndex(tuple(rows), np.array(list(rows.values()), dtype=np.float32))


class TestConstruction:
    def test_reports_its_size_and_dimension(self):
        index = build({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        assert index.count == 2
        assert index.dimension == 2

    def test_rejects_misaligned_ids_and_vectors(self):
        with pytest.raises(VectorIndexError, match="align"):
            NumpyVectorIndex(("a", "b"), np.array([[1.0, 0.0]], dtype=np.float32))

    def test_rejects_duplicate_chunk_ids(self):
        with pytest.raises(VectorIndexError, match="duplicate"):
            NumpyVectorIndex(("a", "a"), np.eye(2, dtype=np.float32))

    def test_rejects_a_zero_vector(self):
        # Cosine similarity against a zero vector is undefined; numpy would
        # return NaN and the ranking would silently become arbitrary.
        with pytest.raises(VectorIndexError, match="zero"):
            NumpyVectorIndex(("a", "b"), np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32))

    def test_rejects_non_finite_values(self):
        with pytest.raises(VectorIndexError, match="finite"):
            NumpyVectorIndex(("a",), np.array([[np.nan, 1.0]], dtype=np.float32))

    def test_rejects_a_non_matrix(self):
        with pytest.raises(VectorIndexError, match="2-D"):
            NumpyVectorIndex(("a",), np.array([1.0, 0.0], dtype=np.float32))


class TestSearch:
    def test_returns_exact_cosine_similarity(self):
        index = build({"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]})
        results = index.search(np.array([1.0, 0.0], dtype=np.float32), 3)

        assert [r.chunk_id for r in results] == ["a", "c", "b"]
        assert results[0].score == pytest.approx(1.0)
        assert results[1].score == pytest.approx(ROOT_HALF)
        assert results[2].score == pytest.approx(0.0, abs=1e-7)

    def test_magnitude_does_not_affect_the_score(self):
        # Cosine, not dot product. The committed cache is float16, so stored
        # norms are 0.9998-1.0001 rather than exactly 1; normalizing here is
        # what makes the score exact anyway.
        index = build({"short": [1.0, 0.0], "long": [400.0, 0.0]})
        results = index.search(np.array([3.0, 0.0], dtype=np.float32), 2)
        assert results[0].score == pytest.approx(1.0)
        assert results[1].score == pytest.approx(1.0)

    def test_truncates_to_top_n(self):
        index = build({"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]})
        assert len(index.search(np.array([1.0, 0.0], dtype=np.float32), 2)) == 2

    def test_ties_break_by_chunk_id_ascending(self):
        index = build({"zulu": [1.0, 0.0], "alpha": [1.0, 0.0], "mike": [1.0, 0.0]})
        results = index.search(np.array([1.0, 0.0], dtype=np.float32), 3)
        assert [r.chunk_id for r in results] == ["alpha", "mike", "zulu"]

    def test_tie_at_the_truncation_boundary_is_deterministic(self):
        # The reason a full stable sort is used instead of argpartition:
        # partition selects *a* top-n set, so which of several tied candidates
        # lands inside the cut is an implementation detail rather than a rule.
        index = build({f"c{i:02d}": [1.0, 0.0] for i in range(20)})
        results = index.search(np.array([1.0, 0.0], dtype=np.float32), 3)
        assert [r.chunk_id for r in results] == ["c00", "c01", "c02"]

    def test_insertion_order_does_not_change_results(self):
        forward = build({"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]})
        backward = build({"c": [1.0, 1.0], "b": [0.0, 1.0], "a": [1.0, 0.0]})
        query = np.array([1.0, 1.0], dtype=np.float32)
        assert forward.search(query, 3) == backward.search(query, 3)

    def test_rejects_a_query_of_the_wrong_dimension(self):
        index = build({"a": [1.0, 0.0]})
        with pytest.raises(VectorIndexError, match="dimension"):
            index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), 1)

    def test_rejects_a_zero_query(self):
        index = build({"a": [1.0, 0.0]})
        with pytest.raises(VectorIndexError, match="zero"):
            index.search(np.zeros(2, dtype=np.float32), 1)

    def test_rejects_a_non_positive_top_n(self):
        index = build({"a": [1.0, 0.0]})
        with pytest.raises(VectorIndexError, match="positive"):
            index.search(np.array([1.0, 0.0], dtype=np.float32), 0)

    def test_accepts_a_2d_query_row(self):
        # Embedders return an (n, dim) matrix; unwrapping a single row at
        # every call site is exactly the kind of chore that gets done wrong.
        index = build({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        results = index.search(np.array([[1.0, 0.0]], dtype=np.float32), 1)
        assert results[0].chunk_id == "a"

    def test_rejects_a_non_finite_query(self):
        index = build({"a": [1.0, 0.0]})
        with pytest.raises(VectorIndexError, match="finite"):
            index.search(np.array([np.inf, 0.0], dtype=np.float32), 1)
