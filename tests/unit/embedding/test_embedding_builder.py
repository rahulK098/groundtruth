"""Building the committed cache from a corpus.

Exercised with a fake embedder, so the build logic is tested without the
`models` extra. The real embedder is a thin adapter over the same Protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np

from groundtruth.config.models import ChunkingConfig, EmbeddingConfig
from groundtruth.corpus.models import Document
from groundtruth.corpus.normalize import normalize_text
from groundtruth.embedding.builder import build_cache, collect_texts
from groundtruth.embedding.cache import CachedOnlyEmbedder
from groundtruth.embedding.store import read_store, verify_store

# The real vendored tokenizer -- chunk boundaries must be realistic.
TOKENIZER = "BAAI/bge-small-en-v1.5"

# A model deliberately absent from KNOWN_MODEL_DIMENSIONS, so an 8-dimensional
# fake is allowed. Using the real bge id here trips EmbeddingConfig's
# dimension check, which is the validator working correctly: 384 is not 8.
MODEL = "test-org/fake-embedder"
REV = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DIM = 8

SENTENCE = (
    "Summary judgment is appropriate when there is no genuine dispute as to any "
    "material fact and the movant is entitled to judgment as a matter of law. "
)


class FakeEmbedder:
    model_id = MODEL
    revision = REV
    dimension = DIM
    normalize = True

    def __init__(self) -> None:
        self.embedded = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.embedded += len(texts)
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = sum(text.encode("utf-8")) or 1
            out[i] = np.array([(seed >> b) % 97 for b in range(DIM)], dtype=np.float32)
        return out


def chunking(size: int, overlap: int) -> ChunkingConfig:
    return ChunkingConfig(chunk_size=size, chunk_overlap=overlap, tokenizer_id=TOKENIZER)


def embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(model_id=MODEL, revision=REV, dimension=DIM, normalize=True)


def docs(n: int = 3) -> tuple[Document, ...]:
    return tuple(Document(doc_id=f"cl-{i}", text=normalize_text(SENTENCE * 6)) for i in range(n))


class TestCollectTexts:
    def test_is_sorted_for_a_stable_diff(self):
        # The cache is committed. Insertion order would make every rebuild
        # produce a noisy, unreviewable diff.
        texts = collect_texts(docs(2), [chunking(64, 16)])
        assert texts == sorted(texts)

    def test_deduplicates_across_documents(self):
        # Identical documents produce identical chunk text; embedding it once
        # is the whole reason keys address text rather than chunk ids.
        one = collect_texts(docs(1), [chunking(64, 16)])
        three = collect_texts(docs(3), [chunking(64, 16)])
        assert one == three

    def test_covers_every_chunking_configuration(self):
        small = set(collect_texts(docs(1), [chunking(64, 16)]))
        both = set(collect_texts(docs(1), [chunking(64, 16), chunking(128, 16)]))
        assert small <= both
        assert len(both) > len(small)

    def test_empty_corpus_yields_nothing(self):
        assert collect_texts((), [chunking(64, 16)]) == []


class TestBuildCache:
    def test_writes_a_verifiable_store(self, tmp_path: Path):
        report = build_cache(
            docs(2), [chunking(64, 16)], embedding_config(), FakeEmbedder(), tmp_path
        )
        assert verify_store(report.directory) == report.total_strings

    def test_covers_every_chunk_string(self, tmp_path: Path):
        # The acceptance criterion for the whole phase: after a build, the
        # cache-only embedder must be able to serve every chunk without a miss.
        documents = docs(2)
        chunkings = [chunking(64, 16), chunking(128, 16)]
        report = build_cache(documents, chunkings, embedding_config(), FakeEmbedder(), tmp_path)
        embedder = CachedOnlyEmbedder(read_store(report.directory))
        texts = collect_texts(documents, chunkings)
        assert embedder.embed(texts).shape == (len(texts), DIM)

    def test_reports_what_it_computed(self, tmp_path: Path):
        report = build_cache(
            docs(1), [chunking(64, 16)], embedding_config(), FakeEmbedder(), tmp_path
        )
        assert report.computed == report.total_strings
        assert report.reused == 0

    def test_a_second_build_reuses_everything(self, tmp_path: Path):
        # Rebuilding after an unrelated change must not re-embed the corpus.
        documents = docs(2)
        chunkings = [chunking(64, 16)]
        build_cache(documents, chunkings, embedding_config(), FakeEmbedder(), tmp_path)

        second = FakeEmbedder()
        report = build_cache(documents, chunkings, embedding_config(), second, tmp_path)
        assert second.embedded == 0
        assert report.computed == 0
        assert report.reused == report.total_strings

    def test_adding_a_chunking_config_only_embeds_the_new_strings(self, tmp_path: Path):
        documents = docs(1)
        build_cache(documents, [chunking(64, 16)], embedding_config(), FakeEmbedder(), tmp_path)

        second = FakeEmbedder()
        report = build_cache(
            documents, [chunking(64, 16), chunking(128, 16)], embedding_config(), second, tmp_path
        )
        assert 0 < second.embedded < report.total_strings

    def test_directory_is_addressed_by_model_and_revision(self, tmp_path: Path):
        report = build_cache(
            docs(1), [chunking(64, 16)], embedding_config(), FakeEmbedder(), tmp_path
        )
        assert "fake-embedder" in report.directory.name
        assert REV[:12] in report.directory.name
        # The organisation prefix must not create a nested directory.
        assert "test-org" not in report.directory.name

    def test_records_build_settings_when_the_embedder_supplies_them(self, tmp_path: Path):
        class Instrumented(FakeEmbedder):
            build_settings: ClassVar[dict[str, object]] = {"device": "cpu", "torch_threads": 1}

        report = build_cache(
            docs(1), [chunking(64, 16)], embedding_config(), Instrumented(), tmp_path
        )
        meta = read_store(report.directory).meta
        assert meta["device"] == "cpu"
        assert meta["torch_threads"] == 1

    def test_respects_the_write_batch_size(self, tmp_path: Path):
        # Batching must not change the result, only the memory profile.
        documents = docs(2)
        a = build_cache(
            documents,
            [chunking(64, 16)],
            embedding_config(),
            FakeEmbedder(),
            tmp_path / "a",
            batch=4,
        )
        b = build_cache(
            documents,
            [chunking(64, 16)],
            embedding_config(),
            FakeEmbedder(),
            tmp_path / "b",
            batch=1000,
        )
        np.testing.assert_array_equal(
            read_store(a.directory).vectors, read_store(b.directory).vectors
        )

    def test_reports_progress(self, tmp_path: Path):
        seen: list[tuple[int, int]] = []
        build_cache(
            docs(2),
            [chunking(64, 16)],
            embedding_config(),
            FakeEmbedder(),
            tmp_path,
            batch=4,
            on_progress=lambda done, total: seen.append((done, total)),
        )
        assert seen
        assert seen[-1][0] == seen[-1][1]
