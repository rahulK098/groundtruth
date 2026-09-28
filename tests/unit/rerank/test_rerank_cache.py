"""The committed rerank-score cache (ADR-0012).

The property under test mirrors the embedding cache's: the gate can serve
every score it needs with no model, and a miss is loud rather than silently
computed or silently zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from groundtruth.rerank.cache import (
    CachedOnlyReranker,
    RecordingReranker,
    RerankCacheMissError,
    passage_key,
    query_key,
    read_rerank_store,
    rerank_store_dir,
    write_rerank_store,
)
from groundtruth.rerank.protocol import RerankScores

MODEL = "BAAI/bge-reranker-base"
REV = "2cfc18c9415c912f9d8155881c133215df768a70"


class LengthReranker:
    """Deterministic stand-in: scores a passage by its length. Counts calls."""

    model_id = MODEL
    revision = REV

    def __init__(self, compute_ms: float = 12.5) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.compute_ms = compute_ms

    def score(self, query: str, passages: Sequence[str]) -> RerankScores:
        self.calls.append((query, tuple(passages)))
        return RerankScores(
            scores=tuple(float(len(p)) for p in passages), compute_ms=self.compute_ms
        )


class TestKeys:
    def test_query_key_is_stable(self):
        assert query_key("q", model_id=MODEL, revision=REV) == query_key(
            "q", model_id=MODEL, revision=REV
        )

    def test_query_key_changes_with_the_model_revision(self):
        # Upgrading weights must never serve scores the old weights produced.
        assert query_key("q", model_id=MODEL, revision=REV) != query_key(
            "q", model_id=MODEL, revision="0" * 40
        )

    def test_query_key_changes_with_the_query_text(self):
        assert query_key("a", model_id=MODEL, revision=REV) != query_key(
            "b", model_id=MODEL, revision=REV
        )

    def test_passage_key_depends_on_text_only(self):
        assert passage_key("same text") == passage_key("same text")
        assert passage_key("same text") != passage_key("other text")


class TestStoreRoundTrip:
    def test_write_then_read_round_trips(self, tmp_path: Path):
        recorder = RecordingReranker(LengthReranker(), None)
        recorder.score("what is X", ["alpha", "beta beta"])
        directory = rerank_store_dir(tmp_path, MODEL, REV)
        write_rerank_store(directory, recorder.store(), build_settings={"device": "cpu"})

        store = read_rerank_store(directory)
        assert store.meta["model_id"] == MODEL
        assert store.meta["revision"] == REV
        assert store.meta["device"] == "cpu"
        assert store.pair_count == 2

    def test_directory_is_addressed_by_model_slug_and_revision(self, tmp_path: Path):
        directory = rerank_store_dir(tmp_path, MODEL, REV)
        assert directory.name == f"bge-reranker-base__{REV[:12]}"

    def test_a_missing_store_is_a_clear_error(self, tmp_path: Path):
        with pytest.raises(RerankCacheMissError, match="gt cache rerank"):
            read_rerank_store(tmp_path / "nowhere")

    def test_written_file_is_sorted_for_a_stable_diff(self, tmp_path: Path):
        recorder = RecordingReranker(LengthReranker(), None)
        for q in ("zeta", "alpha", "mu"):
            recorder.score(q, ["p1 text", "p2 text"])
        directory = rerank_store_dir(tmp_path, MODEL, REV)
        write_rerank_store(directory, recorder.store())

        lines = (directory / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        keys = [line.split('"query_key": "')[1][:32] for line in lines]
        assert keys == sorted(keys)


class TestCachedOnlyReranker:
    def _store(self, tmp_path: Path) -> Path:
        recorder = RecordingReranker(LengthReranker(compute_ms=40.0), None)
        recorder.score("what is X", ["alpha", "beta beta"])
        directory = rerank_store_dir(tmp_path, MODEL, REV)
        write_rerank_store(directory, recorder.store())
        return directory

    def test_serves_recorded_scores_in_passage_order(self, tmp_path: Path):
        reranker = CachedOnlyReranker(read_rerank_store(self._store(tmp_path)))
        result = reranker.score("what is X", ["beta beta", "alpha"])
        assert result.scores == (9.0, 5.0)

    def test_reports_the_recorded_compute_time_not_the_lookup_time(self, tmp_path: Path):
        # ADR-0012: a lookup costs ~0 ms; the latency column must show what
        # the model actually cost, or the reranker tradeoff is unarguable.
        reranker = CachedOnlyReranker(read_rerank_store(self._store(tmp_path)))
        assert reranker.score("what is X", ["alpha"]).compute_ms == 40.0

    def test_exposes_the_store_model_identity(self, tmp_path: Path):
        reranker = CachedOnlyReranker(read_rerank_store(self._store(tmp_path)))
        assert reranker.model_id == MODEL
        assert reranker.revision == REV

    def test_an_unknown_query_raises_naming_the_regeneration_command(self, tmp_path: Path):
        reranker = CachedOnlyReranker(read_rerank_store(self._store(tmp_path)))
        with pytest.raises(RerankCacheMissError, match="gt cache rerank"):
            reranker.score("a query nobody scored", ["alpha"])

    def test_an_unknown_passage_for_a_known_query_raises(self, tmp_path: Path):
        # A chunking change produces new passage text under the same query --
        # that must be a miss, never a silently-absent score.
        reranker = CachedOnlyReranker(read_rerank_store(self._store(tmp_path)))
        with pytest.raises(RerankCacheMissError, match="1 of 2"):
            reranker.score("what is X", ["alpha", "never scored"])

    def test_an_empty_passage_list_returns_nothing(self, tmp_path: Path):
        reranker = CachedOnlyReranker(read_rerank_store(self._store(tmp_path)))
        assert reranker.score("what is X", []).scores == ()


class TestRecordingReranker:
    def test_computes_and_records_a_new_batch(self):
        delegate = LengthReranker()
        recorder = RecordingReranker(delegate, None)
        result = recorder.score("q", ["aa", "bbb"])
        assert result.scores == (2.0, 3.0)
        assert len(delegate.calls) == 1
        assert recorder.computed_queries == 1

    def test_a_fully_cached_batch_is_served_without_calling_the_model(self, tmp_path: Path):
        first = RecordingReranker(LengthReranker(), None)
        first.score("q", ["aa", "bbb"])
        directory = rerank_store_dir(tmp_path, MODEL, REV)
        write_rerank_store(directory, first.store())

        delegate = LengthReranker()
        second = RecordingReranker(delegate, read_rerank_store(directory))
        result = second.score("q", ["bbb", "aa"])
        assert result.scores == (3.0, 2.0)
        assert delegate.calls == []
        assert second.reused_queries == 1

    def test_a_partially_cached_batch_is_recomputed_whole(self, tmp_path: Path):
        # Recomputing the whole batch is what makes the recorded compute_ms
        # the cost of a real batch, not of an arbitrary leftover fragment.
        first = RecordingReranker(LengthReranker(), None)
        first.score("q", ["aa"])
        directory = rerank_store_dir(tmp_path, MODEL, REV)
        write_rerank_store(directory, first.store())

        delegate = LengthReranker()
        second = RecordingReranker(delegate, read_rerank_store(directory))
        second.score("q", ["aa", "cccc"])
        assert delegate.calls == [("q", ("aa", "cccc"))]

    def test_the_merged_store_keeps_earlier_queries(self, tmp_path: Path):
        first = RecordingReranker(LengthReranker(), None)
        first.score("q1", ["aa"])
        directory = rerank_store_dir(tmp_path, MODEL, REV)
        write_rerank_store(directory, first.store())

        second = RecordingReranker(LengthReranker(), read_rerank_store(directory))
        second.score("q2", ["bbb"])
        merged = second.store()
        assert merged.pair_count == 2

    def test_passes_through_the_delegate_identity(self):
        recorder = RecordingReranker(LengthReranker(), None)
        assert recorder.model_id == MODEL
        assert recorder.revision == REV
