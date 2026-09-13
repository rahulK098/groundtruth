"""Embedding cache key derivation and on-disk format.

The cache is committed to the repository, so its format is a public contract:
a reviewer has to be able to see what changed in a diff, and a stale entry has
to be detectable rather than silently served.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from groundtruth.embedding.hashing import embedding_key
from groundtruth.embedding.store import (
    KEYS_FILENAME,
    META_FILENAME,
    VECTORS_FILENAME,
    EmbeddingStoreError,
    read_store,
    store_dir,
    verify_store,
    write_store,
)

MODEL = "BAAI/bge-small-en-v1.5"
REV = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def vectors(n: int, dim: int = 4) -> np.ndarray:
    return np.arange(n * dim, dtype=np.float32).reshape(n, dim) / (n * dim)


class TestKeyDerivation:
    def test_is_deterministic(self):
        a = embedding_key("hello", model_id=MODEL, revision=REV, normalize=True)
        b = embedding_key("hello", model_id=MODEL, revision=REV, normalize=True)
        assert a == b

    def test_differs_by_text(self):
        a = embedding_key("hello", model_id=MODEL, revision=REV, normalize=True)
        b = embedding_key("world", model_id=MODEL, revision=REV, normalize=True)
        assert a != b

    def test_differs_by_model(self):
        a = embedding_key("t", model_id=MODEL, revision=REV, normalize=True)
        b = embedding_key("t", model_id="other/model", revision=REV, normalize=True)
        assert a != b

    def test_differs_by_revision(self):
        # The point of pinning: new weights must not serve old vectors.
        a = embedding_key("t", model_id=MODEL, revision=REV, normalize=True)
        b = embedding_key("t", model_id=MODEL, revision="0" * 40, normalize=True)
        assert a != b

    def test_differs_by_normalization(self):
        a = embedding_key("t", model_id=MODEL, revision=REV, normalize=True)
        b = embedding_key("t", model_id=MODEL, revision=REV, normalize=False)
        assert a != b

    def test_hashes_the_exact_string_sent_to_the_model(self):
        # bge prefixes queries but not passages. Because the key covers the
        # final string, changing the prefix correctly invalidates query
        # vectors and correctly leaves passage vectors alone -- no separate
        # "kind" flag needed.
        prefix = "Represent this sentence for searching relevant passages: "
        bare = embedding_key("summary judgment", model_id=MODEL, revision=REV, normalize=True)
        prefixed = embedding_key(
            prefix + "summary judgment", model_id=MODEL, revision=REV, normalize=True
        )
        assert bare != prefixed

    def test_is_16_hex_characters(self):
        key = embedding_key("t", model_id=MODEL, revision=REV, normalize=True)
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)

    def test_components_are_not_positionally_confusable(self):
        a = embedding_key("t", model_id="ab", revision="c", normalize=True)
        b = embedding_key("t", model_id="a", revision="bc", normalize=True)
        assert a != b


class TestStoreDirectory:
    def test_encodes_model_and_revision(self, tmp_path: Path):
        d = store_dir(tmp_path, MODEL, REV)
        assert "bge-small-en-v1.5" in d.name
        assert REV[:12] in d.name

    def test_different_revisions_get_different_directories(self, tmp_path: Path):
        # Two revisions must coexist rather than overwrite each other.
        assert store_dir(tmp_path, MODEL, REV) != store_dir(tmp_path, MODEL, "0" * 40)

    def test_directory_name_has_no_path_separators(self, tmp_path: Path):
        # "BAAI/bge-..." must not become a nested directory.
        assert "/" not in store_dir(tmp_path, MODEL, REV).name


class TestRoundTrip:
    def test_vectors_survive(self, tmp_path: Path):
        keys = ["k0", "k1", "k2"]
        vecs = vectors(3)
        write_store(tmp_path, keys, vecs, model_id=MODEL, revision=REV, normalize=True)
        loaded = read_store(tmp_path)
        # fp16 storage is lossy by design; compare at fp16 resolution.
        np.testing.assert_allclose(loaded.vectors, vecs, atol=1e-3)

    def test_keys_map_to_the_right_rows(self, tmp_path: Path):
        keys = ["k0", "k1", "k2"]
        vecs = vectors(3)
        write_store(tmp_path, keys, vecs, model_id=MODEL, revision=REV, normalize=True)
        loaded = read_store(tmp_path)
        for i, key in enumerate(keys):
            np.testing.assert_allclose(loaded.vector_for(key), vecs[i], atol=1e-3)

    def test_vectors_load_as_float32(self, tmp_path: Path):
        # Stored as fp16 to halve the committed size, used as fp32 so
        # downstream arithmetic is not silently half precision.
        write_store(tmp_path, ["k"], vectors(1), model_id=MODEL, revision=REV, normalize=True)
        assert read_store(tmp_path).vectors.dtype == np.float32

    def test_stored_on_disk_as_float16(self, tmp_path: Path):
        write_store(tmp_path, ["k"], vectors(1), model_id=MODEL, revision=REV, normalize=True)
        assert np.load(tmp_path / VECTORS_FILENAME).dtype == np.float16

    def test_writes_the_three_expected_files(self, tmp_path: Path):
        write_store(tmp_path, ["k"], vectors(1), model_id=MODEL, revision=REV, normalize=True)
        for name in (VECTORS_FILENAME, KEYS_FILENAME, META_FILENAME):
            assert (tmp_path / name).is_file()

    def test_keys_file_is_line_diffable(self, tmp_path: Path):
        # One JSON object per line, so git can delta it and a reviewer can see
        # exactly which entries changed.
        write_store(
            tmp_path, ["k0", "k1"], vectors(2), model_id=MODEL, revision=REV, normalize=True
        )
        lines = (tmp_path / KEYS_FILENAME).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["key"] == "k0"
        assert json.loads(lines[0])["row"] == 0

    def test_meta_records_the_model_identity(self, tmp_path: Path):
        write_store(tmp_path, ["k"], vectors(1), model_id=MODEL, revision=REV, normalize=True)
        meta = json.loads((tmp_path / META_FILENAME).read_text(encoding="utf-8"))
        assert meta["model_id"] == MODEL
        assert meta["revision"] == REV
        assert meta["dimension"] == 4
        assert meta["count"] == 1

    def test_empty_store_round_trips(self, tmp_path: Path):
        write_store(
            tmp_path,
            [],
            np.zeros((0, 4), dtype=np.float32),
            model_id=MODEL,
            revision=REV,
            normalize=True,
        )
        assert read_store(tmp_path).count == 0

    def test_rejects_a_key_count_mismatch(self, tmp_path: Path):
        with pytest.raises(EmbeddingStoreError, match="rows"):
            write_store(
                tmp_path, ["a", "b"], vectors(3), model_id=MODEL, revision=REV, normalize=True
            )

    def test_rejects_duplicate_keys(self, tmp_path: Path):
        # A duplicate would make vector_for ambiguous and silently return one
        # of two different embeddings.
        with pytest.raises(EmbeddingStoreError, match="duplicate"):
            write_store(
                tmp_path, ["a", "a"], vectors(2), model_id=MODEL, revision=REV, normalize=True
            )


class TestVerification:
    def test_accepts_an_intact_store(self, tmp_path: Path):
        write_store(
            tmp_path, ["k0", "k1"], vectors(2), model_id=MODEL, revision=REV, normalize=True
        )
        assert verify_store(tmp_path) == 2

    def test_detects_truncated_vectors(self, tmp_path: Path):
        # The failure this exists to catch: keys and vectors drifting out of
        # alignment, which would serve the wrong embedding for every row after
        # the break -- plausible numbers, silently wrong.
        write_store(
            tmp_path, ["k0", "k1", "k2"], vectors(3), model_id=MODEL, revision=REV, normalize=True
        )
        np.save(tmp_path / VECTORS_FILENAME, np.zeros((2, 4), dtype=np.float16))
        with pytest.raises(EmbeddingStoreError, match="does not match"):
            verify_store(tmp_path)

    def test_detects_a_dimension_mismatch(self, tmp_path: Path):
        write_store(tmp_path, ["k0"], vectors(1), model_id=MODEL, revision=REV, normalize=True)
        np.save(tmp_path / VECTORS_FILENAME, np.zeros((1, 99), dtype=np.float16))
        with pytest.raises(EmbeddingStoreError, match="dimension"):
            verify_store(tmp_path)

    def test_reports_a_missing_store(self, tmp_path: Path):
        with pytest.raises(EmbeddingStoreError, match="not found"):
            read_store(tmp_path)

    def test_reports_an_unreadable_meta(self, tmp_path: Path):
        write_store(tmp_path, ["k"], vectors(1), model_id=MODEL, revision=REV, normalize=True)
        (tmp_path / META_FILENAME).write_text("{not json", encoding="utf-8")
        with pytest.raises(EmbeddingStoreError, match="parse"):
            read_store(tmp_path)
