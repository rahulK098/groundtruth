"""Request and response types -- separate from the core's models, per api.md.

The core's ``RetrievalResult`` is what the pipeline returns; ``SearchResponse``
is what the HTTP contract promises. Keeping them distinct means a change to
the core's internals cannot silently change the wire format.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from groundtruth.retrieval.models import PassageScores, RetrievalResult, StageLatenciesMs


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    config_name: str = Field(min_length=1)
    #: Defaults to the config's own top_k. Bounded by the config's dense
    #: candidate pool, validated in the route (the bound depends on the config).
    top_k: int | None = Field(default=None, ge=1)


class PassageOut(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int
    rank: int
    #: Per stage, never blended: "the dense arm never surfaced it" and "the
    #: reranker demoted it" are different failures (api.md).
    scores: PassageScores


class SearchResponse(BaseModel):
    config_name: str
    config_hash: str
    passages: list[PassageOut]
    latency_ms: StageLatenciesMs

    @classmethod
    def from_result(cls, result: RetrievalResult) -> SearchResponse:
        return cls(
            config_name=result.config_name,
            config_hash=result.config_hash,
            passages=[PassageOut(**p.model_dump()) for p in result.passages],
            latency_ms=result.latency_ms,
        )


class ConfigSummary(BaseModel):
    name: str
    config_hash: str
    retrieval_mode: str
    reranker: bool
    #: Listed configs are not all served: the reranker arm costs ~105 s per
    #: query on CPU, so it is opt-in (GT_SERVE_CONFIGS), not a default.
    served: bool


class ModelIdentity(BaseModel):
    id: str
    revision: str


class VersionInfo(BaseModel):
    """The four identity components a result's run fingerprint is built from."""

    git_sha: str
    code_version: str
    embedding_model: ModelIdentity
    reranker_model: ModelIdentity | None
    corpus_manifest_hash: str
    golden_set_hash: str
