"""What a retrieval run returns.

These are the values that cross a boundary -- into the scorer, into a result
file, into an HTTP response -- so unlike :class:`~groundtruth.index.models.
ScoredChunk` they are validated pydantic models.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from groundtruth.frozen import FrozenModel


class PassageScores(FrozenModel):
    """Per-stage scores for one retrieved passage.

    Every stage is reported separately, and ``None`` means *this stage did not
    produce a number for this passage* rather than zero. That distinction is
    the entire basis of the error analysis: "the dense arm never surfaced it"
    and "the reranker demoted it" are different failures with different fixes,
    and a single blended score cannot tell them apart.
    """

    #: Cosine similarity, present when the dense arm returned this chunk.
    dense: float | None = None
    #: Lexical score, present when the lexical arm returned this chunk. Not
    #: comparable to ``dense`` -- different scale, different units.
    lexical: float | None = None
    #: Fused score. Present only in hybrid mode: in dense mode there is
    #: nothing to fuse, and reporting a number there would be a fiction.
    fused: float | None = None
    #: Cross-encoder score, present only when reranking ran.
    rerank: float | None = None


class StageLatenciesMs(FrozenModel):
    """Wall-clock cost of each pipeline stage, in milliseconds.

    Broken down per stage because the final recommendation quotes a
    quality-versus-latency tradeoff. A single total would make "the reranker
    costs 240 ms" an assertion rather than a measurement.

    A stage that did not run is 0.0.
    """

    embed: float = Field(default=0.0, ge=0.0)
    dense: float = Field(default=0.0, ge=0.0)
    lexical: float = Field(default=0.0, ge=0.0)
    fuse: float = Field(default=0.0, ge=0.0)
    rerank: float = Field(default=0.0, ge=0.0)
    #: Measured around the whole call, so it includes the assembly work the
    #: individual stages do not cover. It is therefore >= their sum.
    total: float = Field(default=0.0, ge=0.0)


class RetrievedPassage(FrozenModel):
    """One result, carrying everything the scorer and a reader need.

    The character span is not decoration: relevance labels are character spans
    into the normalized document (ADR-0002), and span-to-chunk resolution is
    impossible without it.
    """

    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    #: 1-based, contiguous, and assigned after truncation.
    rank: int = Field(ge=1)
    scores: PassageScores = PassageScores()

    @model_validator(mode="after")
    def _span_must_be_well_formed(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) must be greater than char_start ({self.char_start})"
            )
        return self


class RetrievalResult(FrozenModel):
    """The ranked answer to one query under one configuration.

    Carries the configuration's **content hash**, not only its name. A result
    file naming a config that has since been edited is worse than no result at
    all; the hash is what makes that detectable (ADR-0010).
    """

    query: str
    config_name: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    passages: tuple[RetrievedPassage, ...] = ()
    #: Deliberately excluded from every identity hash: latency is a property
    #: of the machine, not of the experiment, and folding it into the run
    #: fingerprint would make two identical runs look like different ones.
    latency_ms: StageLatenciesMs = StageLatenciesMs()

    @model_validator(mode="after")
    def _ranks_must_be_contiguous_and_chunks_unique(self) -> Self:
        expected = list(range(1, len(self.passages) + 1))
        if [passage.rank for passage in self.passages] != expected:
            raise ValueError(
                "passage ranks must be 1-based and contiguous in order; "
                "every metric downstream reads rank as a position"
            )
        chunk_ids = [passage.chunk_id for passage in self.passages]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError(
                "a chunk appears more than once; it would be counted twice by "
                "Recall and credited twice by nDCG"
            )
        return self
