"""Cache-backed embedders, and the guard that keeps the gate honest."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from groundtruth.embedding.cache import CachedEmbedder, CachedOnlyEmbedder, EmbeddingCacheMissError
from groundtruth.embedding.hashing import embedding_key
from groundtruth.embedding.protocol import Embedder
from groundtruth.embedding.store import read_store, write_store

MODEL = "BAAI/bge-small-en-v1.5"
REV = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DIM = 4


class FakeEmbedder:
    """Deterministic stand-in. Counts calls so caching can be asserted."""

    def __init__(self) -> None:
        self.calls: list[Sequence[str]] = []

    model_id = MODEL
    revision = REV
    dimension = DIM
    normalize = True

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        rng = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = sum(text.encode("utf-8")) or 1
            rng[i] = np.array([seed % 7, seed % 11, seed % 13, seed % 17], dtype=np.float32)
        return rng

    @property
    def embedded_count(self) -> int:
        return sum(len(c) for c in self.calls)


def build_store(tmp_path: Path, texts: Sequence[str]):
    fake = FakeEmbedder()
    keys = [embedding_key(t, model_id=MODEL, revision=REV, normalize=True) for t in texts]
    write_store(tmp_path, keys, fake.embed(texts), model_id=MODEL, revision=REV, normalize=True)
    return read_store(tmp_path)


class TestCachedOnly:
    def test_serves_committed_vectors(self, tmp_path: Path):
        store = build_store(tmp_path, ["alpha", "beta"])
        out = CachedOnlyEmbedder(store).embed(["alpha", "beta"])
        assert out.shape == (2, DIM)

    def test_rows_align_with_the_requested_order(self, tmp_path: Path):
        store = build_store(tmp_path, ["alpha", "beta"])
        embedder = CachedOnlyEmbedder(store)
        forward = embedder.embed(["alpha", "beta"])
        reverse = embedder.embed(["beta", "alpha"])
        np.testing.assert_array_equal(forward[0], reverse[1])
        np.testing.assert_array_equal(forward[1], reverse[0])

    def test_raises_on_a_miss(self, tmp_path: Path):
        # The whole point. A silent fallback would mix fresh and stale vectors
        # and report confident, wrong numbers.
        store = build_store(tmp_path, ["alpha"])
        with pytest.raises(EmbeddingCacheMissError):
            CachedOnlyEmbedder(store).embed(["never embedded"])

    def test_miss_message_names_the_counts(self, tmp_path: Path):
        store = build_store(tmp_path, ["alpha"])
        with pytest.raises(EmbeddingCacheMissError) as exc:
            CachedOnlyEmbedder(store).embed(["alpha", "missing one", "missing two"])
        assert "2 of 3" in str(exc.value)

    def test_miss_message_names_the_fix(self, tmp_path: Path):
        # An error saying only "cache miss" sends someone reading source.
        store = build_store(tmp_path, ["alpha"])
        with pytest.raises(EmbeddingCacheMissError) as exc:
            CachedOnlyEmbedder(store).embed(["missing"])
        assert "gt cache build" in str(exc.value)

    def test_takes_identity_from_the_store(self, tmp_path: Path):
        embedder = CachedOnlyEmbedder(build_store(tmp_path, ["alpha"]))
        assert embedder.model_id == MODEL
        assert embedder.revision == REV
        assert embedder.dimension == DIM

    def test_empty_input_returns_an_empty_matrix(self, tmp_path: Path):
        out = CachedOnlyEmbedder(build_store(tmp_path, ["alpha"])).embed([])
        assert out.shape == (0, DIM)

    def test_satisfies_the_embedder_protocol(self, tmp_path: Path):
        assert isinstance(CachedOnlyEmbedder(build_store(tmp_path, ["a"])), Embedder)


class TestCachedWithFallback:
    def test_computes_what_is_missing(self, tmp_path: Path):
        store = build_store(tmp_path, ["alpha"])
        fake = FakeEmbedder()
        out = CachedEmbedder(store, fake).embed(["alpha", "brand new"])
        assert out.shape == (2, DIM)
        assert fake.embedded_count == 1  # only the miss

    def test_does_not_recompute_a_hit(self, tmp_path: Path):
        store = build_store(tmp_path, ["alpha", "beta"])
        fake = FakeEmbedder()
        CachedEmbedder(store, fake).embed(["alpha", "beta"])
        assert fake.embedded_count == 0

    def test_records_computed_vectors_for_write_back(self, tmp_path: Path):
        embedder = CachedEmbedder(build_store(tmp_path, []), FakeEmbedder())
        embedder.embed(["fresh"])
        assert len(embedder.computed) == 1

    def test_reuses_a_vector_computed_earlier_in_the_session(self, tmp_path: Path):
        fake = FakeEmbedder()
        embedder = CachedEmbedder(None, fake)
        embedder.embed(["repeated"])
        embedder.embed(["repeated"])
        assert fake.embedded_count == 1

    def test_deduplicates_within_one_call(self, tmp_path: Path):
        # Legal corpora repeat boilerplate constantly; embedding the same
        # string twice in one batch is pure cost.
        fake = FakeEmbedder()
        out = CachedEmbedder(None, fake).embed(["same", "same", "same"])
        assert fake.embedded_count == 1
        np.testing.assert_array_equal(out[0], out[2])

    def test_works_with_no_existing_store(self, tmp_path: Path):
        out = CachedEmbedder(None, FakeEmbedder()).embed(["a", "b"])
        assert out.shape == (2, DIM)

    def test_empty_input_returns_an_empty_matrix(self):
        assert CachedEmbedder(None, FakeEmbedder()).embed([]).shape == (0, DIM)


class TestImportHygiene:
    """The structural guarantee behind ADR-0003."""

    def test_the_cache_path_does_not_import_torch(self):
        # torch and sentence-transformers live in an optional extra the gate
        # never installs, so a download during a gate run is impossible rather
        # than merely discouraged. This asserts the lazy-import discipline that
        # keeps it that way.
        assert "torch" not in sys.modules
        assert "sentence_transformers" not in sys.modules
