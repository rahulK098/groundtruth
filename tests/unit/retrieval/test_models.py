"""The result value objects.

Their validators exist to make a malformed result impossible to write to disk.
A result file is read by the scorer, the gate and the report, none of which
re-check these properties -- so anything that gets past here is believed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from groundtruth.retrieval.models import (
    PassageScores,
    RetrievalResult,
    RetrievedPassage,
    StageLatenciesMs,
)


def passage(chunk_id: str = "c1", rank: int = 1, **kwargs: object) -> RetrievedPassage:
    fields: dict[str, object] = {
        "chunk_id": chunk_id,
        "doc_id": "doc-1",
        "text": "summary judgment",
        "char_start": 0,
        "char_end": 16,
        "rank": rank,
    }
    fields.update(kwargs)
    return RetrievedPassage(**fields)  # type: ignore[arg-type]


class TestRetrievedPassage:
    def test_rejects_an_inverted_span(self):
        with pytest.raises(ValidationError, match="char_end"):
            passage(char_start=100, char_end=10)

    def test_rejects_rank_zero(self):
        # Ranks are 1-based everywhere: MRR divides by them.
        with pytest.raises(ValidationError):
            passage(rank=0)

    def test_defaults_to_no_scores_at_all(self):
        assert passage().scores == PassageScores()


class TestRetrievalResult:
    def test_accepts_a_well_formed_ranking(self):
        result = RetrievalResult(
            query="q",
            config_name="dense_512",
            config_hash="abc123",
            passages=(passage("c1", 1), passage("c2", 2)),
        )
        assert len(result.passages) == 2

    def test_rejects_non_contiguous_ranks(self):
        with pytest.raises(ValidationError, match="contiguous"):
            RetrievalResult(
                query="q",
                config_name="dense_512",
                config_hash="abc123",
                passages=(passage("c1", 1), passage("c2", 3)),
            )

    def test_rejects_ranks_out_of_order(self):
        with pytest.raises(ValidationError, match="contiguous"):
            RetrievalResult(
                query="q",
                config_name="dense_512",
                config_hash="abc123",
                passages=(passage("c1", 2), passage("c2", 1)),
            )

    def test_rejects_a_repeated_chunk(self):
        # Recall would count one chunk twice and nDCG would credit it twice.
        with pytest.raises(ValidationError, match="more than once"):
            RetrievalResult(
                query="q",
                config_name="dense_512",
                config_hash="abc123",
                passages=(passage("c1", 1), passage("c1", 2)),
            )

    def test_an_empty_ranking_is_valid(self):
        # A query that retrieves nothing is a finding, not a crash.
        result = RetrievalResult(query="q", config_name="dense_512", config_hash="abc123")
        assert result.passages == ()
        assert result.latency_ms == StageLatenciesMs()
