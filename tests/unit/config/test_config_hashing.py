"""Canonicalization and content hashing.

These functions decide whether two evaluation runs are comparable, so their
behaviour is pinned hard here. A silent change to canonicalization would
orphan every committed result without any test failing elsewhere.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from groundtruth.config.hashing import canonical_json, content_hash, run_fingerprint


class TestCanonicalJson:
    def test_sorts_keys_regardless_of_insertion_order(self):
        assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})

    def test_output_is_compact(self):
        # No spaces after separators: whitespace would change the hash for no
        # semantic reason.
        assert canonical_json({"a": 1, "b": 2}) == '{"a":1,"b":2}'

    def test_preserves_array_order(self):
        # Key order is arbitrary and must be normalized. List order is
        # meaningful (it is the order the pipeline runs in) and must not be.
        assert canonical_json({"k": [3, 1, 2]}) == '{"k":[3,1,2]}'

    def test_does_not_escape_non_ascii(self):
        # ensure_ascii=False. Escaping would make the canonical form depend on
        # the encoder rather than the content.
        assert canonical_json({"k": "café"}) == '{"k":"café"}'

    def test_nested_keys_are_sorted_at_every_depth(self):
        nested = {"outer": {"z": 1, "a": {"y": 2, "b": 3}}}
        assert canonical_json(nested) == '{"outer":{"a":{"b":3,"y":2},"z":1}}'


class TestContentHash:
    def test_is_deterministic(self):
        payload = {"a": 1, "b": [2, 3]}
        assert content_hash(payload) == content_hash(payload)

    def test_is_insensitive_to_key_order(self):
        assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})

    def test_is_sensitive_to_values(self):
        assert content_hash({"a": 1}) != content_hash({"a": 2})

    def test_is_sensitive_to_list_order(self):
        assert content_hash({"a": [1, 2]}) != content_hash({"a": [2, 1]})

    def test_default_digest_is_16_hex_chars(self):
        # digest_size=8 bytes. Short enough to read in a filename, long enough
        # that an accidental collision across a handful of configs is absurd.
        digest = content_hash({"a": 1})
        assert len(digest) == 16
        assert all(c in "0123456789abcdef" for c in digest)

    def test_empty_payload_is_hashable(self):
        assert len(content_hash({})) == 16

    def test_known_answer_is_locked(self):
        # Locks the whole canonicalize-then-blake2b chain. If this changes,
        # every committed result is orphaned -- which must be a deliberate act
        # accompanied by a re-baseline, not a silent refactor.
        # Derived independently with the standard library alone
        # (json.dumps -> blake2b, digest_size=8), not copied from this
        # implementation's output, so the lock is a real check rather than a
        # restatement of whatever the code happens to do.
        assert content_hash({"a": 1, "b": "x"}) == "a18952d1cd646943"


class TestRunFingerprint:
    COMPONENTS: ClassVar[dict[str, str]] = {
        "config_hash": "aaaaaaaaaaaaaaaa",
        "golden_set_hash": "sha256:bbbb",
        "corpus_manifest_hash": "sha256:cccc",
        "code_version": "0.1.0",
    }

    def test_is_deterministic(self):
        assert run_fingerprint(**self.COMPONENTS) == run_fingerprint(**self.COMPONENTS)

    @pytest.mark.parametrize("component", sorted(COMPONENTS))
    def test_changes_when_any_component_changes(self, component: str):
        # The whole point: a result is only comparable to another if all four
        # axes match. Each must therefore move the fingerprint.
        mutated = {**self.COMPONENTS, component: "CHANGED"}
        assert run_fingerprint(**mutated) != run_fingerprint(**self.COMPONENTS)

    def test_components_are_not_positionally_confusable(self):
        # Naive concatenation would make ("ab","c") and ("a","bc") collide.
        # A delimiter alone is not enough if it can appear inside a component.
        a = run_fingerprint(
            config_hash="ab",
            golden_set_hash="c",
            corpus_manifest_hash="d",
            code_version="e",
        )
        b = run_fingerprint(
            config_hash="a",
            golden_set_hash="bc",
            corpus_manifest_hash="d",
            code_version="e",
        )
        assert a != b
