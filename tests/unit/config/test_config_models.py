"""RetrievalConfig validation and identity.

Two jobs here. First, reject configurations that are internally incoherent --
especially ones carrying settings the chosen mode ignores, because dead config
is a landmine (someone tunes a value that has no effect and concludes the
knob does nothing). Second, pin config identity, since every result file is
addressed by it.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from groundtruth.config.models import RetrievalConfig


def dense_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "dense_512",
        "description": "Baseline: dense retrieval over 512-token chunks.",
        "schema_version": 1,
        "retrieval_mode": "dense",
        "top_k": 10,
        "dense_top_n": 50,
        "chunking": {
            "strategy": "fixed_token_window",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "tokenizer_id": "BAAI/bge-small-en-v1.5",
        },
        "embedding": {
            "model_id": "BAAI/bge-small-en-v1.5",
            # Deliberately not a real-looking SHA: a plausible 40-hex string
            # here would eventually be copied into a real config as if it were
            # the pinned revision.
            "revision": "fixture-revision",
            "dimension": 384,
            "normalize": True,
            "query_prefix": "Represent this sentence for searching relevant passages: ",
        },
        "lexical": None,
        "fusion": None,
        "reranker": {"enabled": False, "model_id": None, "revision": None, "top_n_in": 50},
    }
    return {**base, **overrides}


def hybrid_payload(**overrides: Any) -> dict[str, Any]:
    # Overrides are applied after the hybrid defaults, not passed through to
    # dense_payload -- otherwise overriding `lexical` or `fusion` collides with
    # the values set here.
    payload = dense_payload(
        name="hybrid_512",
        retrieval_mode="hybrid",
        lexical={"backend": "bm25", "top_n": 50, "language": "english", "k1": 1.2, "b": 0.75},
        fusion={"method": "rrf", "k": 60},
    )
    return {**payload, **overrides}


class TestConstruction:
    def test_dense_config_is_valid(self):
        cfg = RetrievalConfig.model_validate(dense_payload())
        assert cfg.retrieval_mode == "dense"
        assert cfg.lexical is None

    def test_hybrid_config_is_valid(self):
        cfg = RetrievalConfig.model_validate(hybrid_payload())
        assert cfg.lexical is not None
        assert cfg.fusion is not None

    def test_config_is_frozen(self):
        cfg = RetrievalConfig.model_validate(dense_payload())
        with pytest.raises(ValidationError):
            cfg.top_k = 5  # type: ignore[misc]

    def test_unknown_field_is_rejected(self):
        # extra="forbid". A typo'd key must fail loudly rather than be ignored,
        # because an ignored key looks exactly like a knob that does nothing.
        with pytest.raises(ValidationError):
            RetrievalConfig.model_validate(dense_payload(chunk_size=256))


class TestCoherenceValidators:
    def test_overlap_must_be_smaller_than_chunk_size(self):
        payload = dense_payload()
        payload["chunking"] = {**payload["chunking"], "chunk_overlap": 512}
        with pytest.raises(ValidationError, match="chunk_overlap"):
            RetrievalConfig.model_validate(payload)

    def test_dense_mode_rejects_lexical_settings(self):
        # Dead config is the trap: a dense config carrying lexical settings
        # invites someone to tune a value the pipeline never reads.
        payload = dense_payload(
            lexical={"backend": "bm25", "top_n": 50, "language": "english", "k1": 1.2, "b": 0.75}
        )
        with pytest.raises(ValidationError, match="dense"):
            RetrievalConfig.model_validate(payload)

    def test_dense_mode_rejects_fusion_settings(self):
        with pytest.raises(ValidationError, match="dense"):
            RetrievalConfig.model_validate(dense_payload(fusion={"method": "rrf", "k": 60}))

    def test_hybrid_mode_requires_lexical(self):
        with pytest.raises(ValidationError, match="hybrid"):
            RetrievalConfig.model_validate(hybrid_payload(lexical=None))

    def test_hybrid_mode_requires_fusion(self):
        with pytest.raises(ValidationError, match="hybrid"):
            RetrievalConfig.model_validate(hybrid_payload(fusion=None))

    def test_enabled_reranker_requires_model_id(self):
        payload = dense_payload(
            reranker={"enabled": True, "model_id": None, "revision": None, "top_n_in": 50}
        )
        with pytest.raises(ValidationError, match="model_id"):
            RetrievalConfig.model_validate(payload)

    def test_enabled_reranker_requires_pinned_revision(self):
        # An unpinned revision means the weights can change under you, which
        # would silently move every metric.
        payload = dense_payload(
            reranker={
                "enabled": True,
                "model_id": "BAAI/bge-reranker-base",
                "revision": None,
                "top_n_in": 50,
            }
        )
        with pytest.raises(ValidationError, match="revision"):
            RetrievalConfig.model_validate(payload)

    def test_reranker_input_must_not_be_smaller_than_output(self):
        payload = dense_payload(
            top_k=10,
            reranker={
                "enabled": True,
                "model_id": "BAAI/bge-reranker-base",
                "revision": "abc123",
                "top_n_in": 5,
            },
        )
        with pytest.raises(ValidationError, match="top_n_in"):
            RetrievalConfig.model_validate(payload)

    def test_dense_candidate_pool_must_not_be_smaller_than_top_k(self):
        with pytest.raises(ValidationError, match="dense_top_n"):
            RetrievalConfig.model_validate(dense_payload(top_k=50, dense_top_n=10))

    def test_embedding_dimension_must_match_the_known_model(self):
        # Catches a typo before twenty minutes of embedding are wasted.
        payload = dense_payload()
        payload["embedding"] = {**payload["embedding"], "dimension": 768}
        with pytest.raises(ValidationError, match="dimension"):
            RetrievalConfig.model_validate(payload)


class TestConfigHash:
    def test_is_deterministic(self):
        assert (
            RetrievalConfig.model_validate(dense_payload()).config_hash
            == RetrievalConfig.model_validate(dense_payload()).config_hash
        )

    @pytest.mark.parametrize("field", ["name", "description"])
    def test_ignores_cosmetic_fields(self, field: str):
        # Renaming a config, or fixing a typo in its description, must not
        # orphan its committed results.
        baseline = RetrievalConfig.model_validate(dense_payload())
        renamed = RetrievalConfig.model_validate(dense_payload(**{field: "something else"}))
        assert baseline.config_hash == renamed.config_hash

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("schema_version", 2),
            ("retrieval_mode", "hybrid"),
            ("top_k", 5),
            ("dense_top_n", 100),
        ],
    )
    def test_changes_for_every_substantive_top_level_field(self, field: str, value: Any):
        baseline = RetrievalConfig.model_validate(dense_payload())
        if field == "retrieval_mode":
            mutated = RetrievalConfig.model_validate(hybrid_payload())
        else:
            mutated = RetrievalConfig.model_validate(dense_payload(**{field: value}))
        assert mutated.config_hash != baseline.config_hash

    @pytest.mark.parametrize(
        ("section", "field", "value"),
        [
            ("chunking", "chunk_size", 256),
            ("chunking", "chunk_overlap", 64),
            ("chunking", "tokenizer_id", "other/tokenizer"),
            ("embedding", "revision", "0000000000000000000000000000000000000000"),
            ("embedding", "normalize", False),
            ("embedding", "query_prefix", ""),
        ],
    )
    def test_changes_for_every_substantive_nested_field(self, section: str, field: str, value: Any):
        baseline = RetrievalConfig.model_validate(dense_payload())
        payload = dense_payload()
        payload[section] = {**payload[section], field: value}
        assert RetrievalConfig.model_validate(payload).config_hash != baseline.config_hash

    def test_spelling_out_a_default_does_not_change_the_hash(self):
        # model_dump without exclude_unset, so two YAML files that differ only
        # in whether they spell out a default are the same configuration.
        payload = dense_payload()
        explicit = RetrievalConfig.model_validate(payload)

        payload_implicit = dense_payload()
        del payload_implicit["embedding"]["normalize"]  # defaults to True
        implicit = RetrievalConfig.model_validate(payload_implicit)

        assert explicit.config_hash == implicit.config_hash

    def test_hash_is_locked(self):
        # Regression lock. Any accidental change to field order, defaults, or
        # the exclusion set breaks this loudly instead of silently orphaning
        # every committed result.
        # Cross-checked: hashing the raw fixture dict (minus name/description)
        # with the standard library alone yields the same value, which also
        # proves model_dump(mode="json") round-trips the payload without
        # coercing any value.
        assert RetrievalConfig.model_validate(dense_payload()).config_hash == "ecf86e9e635bfa6d"
