"""Retrieval configuration.

A configuration is the independent variable of every experiment this project
runs, so it is frozen, validated hard, and addressed by content.

The validators exist to reject configurations that are *internally incoherent*,
not merely out of range. The most important of those rules is that a config may
not carry settings its mode ignores: dead config is a landmine, because someone
eventually tunes a value the pipeline never reads and concludes the knob does
nothing.
"""

from __future__ import annotations

from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from groundtruth.config.hashing import content_hash

#: Bump deliberately, in its own commit, when the schema changes in a way that
#: should invalidate existing baselines. It is part of the config hash, so a
#: bump re-addresses every result -- which is the intended effect.
SCHEMA_VERSION: Final[int] = 1

#: Guards against a typo costing twenty minutes of embedding before failing.
#: A model absent from this table is allowed through unchecked rather than
#: rejected, so an unfamiliar model is usable without editing this file.
KNOWN_MODEL_DIMENSIONS: Final[dict[str, int]] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}


class _Frozen(BaseModel):
    """Immutable, strict base.

    ``extra="forbid"`` matters more than it looks: a silently ignored key is
    indistinguishable from a knob that does nothing.

    ``protected_namespaces=()`` disables pydantic's warning about fields
    beginning with ``model_``. Here ``model_id`` means a retrieval model, and
    renaming it to dodge a framework warning would be the tail wagging the dog.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class ChunkingConfig(_Frozen):
    """How documents are split. The independent variable of the chunk-size arm."""

    strategy: Literal["fixed_token_window"] = "fixed_token_window"
    chunk_size: int = Field(ge=64, le=2048, description="Window width, in tokens.")
    chunk_overlap: int = Field(ge=0, description="Overlap between windows, in tokens.")
    tokenizer_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _overlap_must_be_smaller_than_the_window(self) -> Self:
        # Equal would produce zero stride and therefore infinite chunks.
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}); stride would be <= 0"
            )
        return self


class EmbeddingConfig(_Frozen):
    """The dense encoder. ``revision`` is pinned so weights cannot drift."""

    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    normalize: bool = True
    #: bge models expect an instruction prefix on queries but NOT on passages.
    #: It is part of the cache key, so changing it correctly invalidates query
    #: vectors while leaving passage vectors alone.
    query_prefix: str = ""

    @model_validator(mode="after")
    def _dimension_must_match_a_known_model(self) -> Self:
        expected = KNOWN_MODEL_DIMENSIONS.get(self.model_id)
        if expected is not None and self.dimension != expected:
            raise ValueError(
                f"dimension {self.dimension} does not match the known dimension "
                f"{expected} for {self.model_id}"
            )
        return self


class LexicalConfig(_Frozen):
    """The lexical arm of hybrid retrieval.

    ``pg_fts`` is Postgres ``tsvector`` ranking. It is deliberately NOT named
    bm25: it has no IDF term at all. See ADR-0007.
    """

    backend: Literal["pg_fts", "bm25"]
    top_n: int = Field(gt=0)
    language: str = "english"
    k1: float = Field(default=1.2, gt=0, description="BM25 term-frequency saturation.")
    b: float = Field(default=0.75, ge=0, le=1, description="BM25 length normalization.")


class FusionConfig(_Frozen):
    """Reciprocal Rank Fusion.

    RRF consumes ranks, not scores, so the dense and lexical arms need no score
    calibration -- which is exactly why it is used here over weighted score
    normalization across two incomparable scales.
    """

    method: Literal["rrf"] = "rrf"
    k: int = Field(default=60, gt=0)


class RerankerConfig(_Frozen):
    """Optional cross-encoder reranking stage."""

    enabled: bool = False
    model_id: str | None = None
    revision: str | None = None
    top_n_in: int = Field(default=50, gt=0, description="Candidates fed to the reranker.")

    @model_validator(mode="after")
    def _enabled_reranker_must_be_fully_pinned(self) -> Self:
        if not self.enabled:
            return self
        if not self.model_id:
            raise ValueError("reranker.model_id is required when the reranker is enabled")
        if not self.revision:
            # An unpinned revision means weights can change underneath a
            # committed baseline, silently moving every metric.
            raise ValueError("reranker.revision must be pinned when the reranker is enabled")
        return self


class RetrievalConfig(_Frozen):
    """One complete, reproducible retrieval configuration."""

    name: str = Field(min_length=1)
    description: str = ""
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    retrieval_mode: Literal["dense", "hybrid"]
    top_k: int = Field(gt=0, description="Results returned and scored.")
    dense_top_n: int = Field(gt=0, description="Candidates drawn from the dense index.")

    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    lexical: LexicalConfig | None = None
    fusion: FusionConfig | None = None
    reranker: RerankerConfig = RerankerConfig()

    @model_validator(mode="after")
    def _mode_must_match_its_components(self) -> Self:
        if self.retrieval_mode == "dense":
            if self.lexical is not None:
                raise ValueError(
                    "retrieval_mode 'dense' must not carry lexical settings: "
                    "the pipeline never reads them, so they are dead config"
                )
            if self.fusion is not None:
                raise ValueError(
                    "retrieval_mode 'dense' must not carry fusion settings: "
                    "there is nothing to fuse"
                )
            return self

        if self.lexical is None:
            raise ValueError("retrieval_mode 'hybrid' requires lexical settings")
        if self.fusion is None:
            raise ValueError("retrieval_mode 'hybrid' requires fusion settings")
        return self

    @model_validator(mode="after")
    def _candidate_pools_must_be_large_enough(self) -> Self:
        # Retrieving fewer candidates than results is silently truncating
        # output, which shows up as an inexplicably poor metric.
        if self.dense_top_n < self.top_k:
            raise ValueError(f"dense_top_n ({self.dense_top_n}) must be >= top_k ({self.top_k})")
        if self.reranker.enabled and self.reranker.top_n_in < self.top_k:
            raise ValueError(
                f"reranker.top_n_in ({self.reranker.top_n_in}) must be >= top_k ({self.top_k})"
            )
        return self

    @property
    def config_hash(self) -> str:
        """Content address of this configuration.

        ``name`` and ``description`` are excluded so renaming a config, or
        fixing a typo in its prose, does not orphan its committed results.

        ``model_dump`` is used *without* ``exclude_unset``, so two YAML files
        differing only in whether they spell out a default hash identically --
        they describe the same experiment.
        """
        payload = self.model_dump(mode="json", exclude={"name", "description"})
        return content_hash(payload)
