"""Corpus snapshot round-tripping and verification.

The snapshot is what makes offline reproduction possible, and its manifest is
an anti-cheat control: the gate hard-fails when the corpus changes, so these
checks have to actually catch a changed corpus.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from groundtruth.corpus.models import CorpusManifest, Document
from groundtruth.corpus.normalize import NORMALIZER_VERSION
from groundtruth.corpus.snapshot import (
    MANIFEST_FILENAME,
    SNAPSHOT_FILENAME,
    SnapshotError,
    content_sha256,
    read_manifest,
    read_snapshot,
    serialize_documents,
    verify_snapshot,
    write_manifest,
    write_snapshot,
)


def docs(n: int = 3) -> tuple[Document, ...]:
    return tuple(
        Document(
            doc_id=f"cl-{i}",
            text=f"Opinion number {i}.",
            title=f"Case {i}",
            court="scotus",
            date_filed="2001-01-01",
            source_url=f"https://example.test/{i}",
        )
        for i in range(n)
    )


def snapshot(tmp_path: Path, documents: tuple[Document, ...]) -> CorpusManifest:
    return write_snapshot(documents, tmp_path, corpus_id="test", source="unit-test")


class TestSerialization:
    def test_is_independent_of_input_order(self):
        # Re-fetching returns documents in a different order. If that changed
        # the corpus hash, the gate would fail for no real reason.
        forward = docs(4)
        assert serialize_documents(forward) == serialize_documents(tuple(reversed(forward)))

    def test_emits_one_line_per_document(self):
        payload = serialize_documents(docs(3)).decode("utf-8")
        assert len([ln for ln in payload.splitlines() if ln.strip()]) == 3

    def test_uses_lf_line_endings(self):
        # Developed on Windows, read in Linux containers. A CRLF here would
        # change the corpus hash between platforms.
        assert b"\r" not in serialize_documents(docs(3))

    def test_empty_corpus_serializes_to_empty_bytes(self):
        assert serialize_documents(()) == b""

    def test_does_not_escape_non_ascii(self):
        doc = Document(doc_id="a", text="café")
        assert "café" in serialize_documents((doc,)).decode("utf-8")


class TestRoundTrip:
    def test_documents_survive_a_write_read_cycle(self, tmp_path: Path):
        original = docs(5)
        snapshot(tmp_path, original)
        assert read_snapshot(tmp_path) == tuple(sorted(original, key=lambda d: d.doc_id))

    def test_manifest_survives_a_write_read_cycle(self, tmp_path: Path):
        written = snapshot(tmp_path, docs(3))
        assert read_manifest(tmp_path) == written

    def test_writes_both_expected_files(self, tmp_path: Path):
        snapshot(tmp_path, docs(2))
        assert (tmp_path / SNAPSHOT_FILENAME).is_file()
        assert (tmp_path / MANIFEST_FILENAME).is_file()

    def test_compression_actually_shrinks_a_repetitive_corpus(self, tmp_path: Path):
        many = tuple(Document(doc_id=f"d{i}", text="the same sentence. " * 50) for i in range(50))
        snapshot(tmp_path, many)
        raw = len(serialize_documents(many))
        compressed = (tmp_path / SNAPSHOT_FILENAME).stat().st_size
        assert compressed < raw

    def test_creates_the_directory_if_absent(self, tmp_path: Path):
        target = tmp_path / "nested" / "corpus"
        write_snapshot(docs(1), target, corpus_id="t", source="s")
        assert (target / SNAPSHOT_FILENAME).is_file()

    def test_empty_corpus_round_trips(self, tmp_path: Path):
        write_snapshot((), tmp_path, corpus_id="t", source="s")
        assert read_snapshot(tmp_path) == ()


class TestManifest:
    def test_records_the_document_count(self, tmp_path: Path):
        assert snapshot(tmp_path, docs(7)).doc_count == 7

    def test_records_the_normalizer_version(self, tmp_path: Path):
        assert snapshot(tmp_path, docs(1)).normalizer_version == NORMALIZER_VERSION

    def test_content_hash_matches_the_documents(self, tmp_path: Path):
        documents = docs(3)
        assert snapshot(tmp_path, documents).content_sha256 == content_sha256(documents)

    def test_hash_identifies_the_uncompressed_form(self, tmp_path: Path):
        # Deliberate: a zstd version or level change must NOT look like a
        # changed corpus to the gate.
        documents = docs(3)
        manifest = snapshot(tmp_path, documents)
        compressed = (tmp_path / SNAPSHOT_FILENAME).read_bytes()
        assert manifest.content_sha256 != hashlib.sha256(compressed).hexdigest()
        assert manifest.content_sha256 == hashlib.sha256(serialize_documents(documents)).hexdigest()

    def test_manifest_hash_is_stable(self, tmp_path: Path):
        manifest = snapshot(tmp_path, docs(2))
        assert manifest.manifest_hash == read_manifest(tmp_path).manifest_hash

    def test_manifest_hash_changes_with_content(self, tmp_path: Path):
        a = write_snapshot(docs(2), tmp_path / "a", corpus_id="t", source="s")
        b = write_snapshot(docs(3), tmp_path / "b", corpus_id="t", source="s")
        assert a.manifest_hash != b.manifest_hash

    def test_manifest_hash_is_prefixed(self, tmp_path: Path):
        assert snapshot(tmp_path, docs(1)).manifest_hash.startswith("sha256:")


class TestVerification:
    def test_accepts_an_intact_snapshot(self, tmp_path: Path):
        original = docs(4)
        snapshot(tmp_path, original)
        assert len(verify_snapshot(tmp_path)) == 4

    def test_detects_tampered_content(self, tmp_path: Path):
        # The anti-cheat check: swapping documents while keeping the manifest
        # must be caught.
        snapshot(tmp_path, docs(3))
        original = read_manifest(tmp_path)

        # Replace the data with a different corpus, then put the original
        # manifest back -- exactly what swapping the corpus while claiming it
        # is unchanged would look like on disk.
        write_snapshot(
            (*docs(3), Document(doc_id="cl-99", text="added")),
            tmp_path,
            corpus_id="t",
            source="s",
        )
        write_manifest(original, tmp_path)

        with pytest.raises(SnapshotError, match="content hash mismatch"):
            verify_snapshot(tmp_path)

    def test_detects_a_normalizer_version_change(self, tmp_path: Path):
        # The subtlest failure mode: text re-normalized under different rules,
        # so every golden-set character offset is silently wrong.
        documents = docs(2)
        written = snapshot(tmp_path, documents)
        stale = written.model_copy(update={"normalizer_version": "999"})
        write_manifest(stale, tmp_path)
        with pytest.raises(SnapshotError, match="NORMALIZER_VERSION"):
            verify_snapshot(tmp_path)

    def test_reports_a_missing_snapshot(self, tmp_path: Path):
        with pytest.raises(SnapshotError, match="not found"):
            read_snapshot(tmp_path)

    def test_reports_a_missing_manifest(self, tmp_path: Path):
        with pytest.raises(SnapshotError, match="not found"):
            read_manifest(tmp_path)

    def test_reports_a_corrupt_archive(self, tmp_path: Path):
        (tmp_path / SNAPSHOT_FILENAME).write_bytes(b"this is not zstd")
        with pytest.raises(SnapshotError, match="decompress"):
            read_snapshot(tmp_path)

    def test_reports_an_unparseable_manifest(self, tmp_path: Path):
        (tmp_path / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")
        with pytest.raises(SnapshotError, match="parse"):
            read_manifest(tmp_path)

    def test_reports_an_invalid_manifest(self, tmp_path: Path):
        (tmp_path / MANIFEST_FILENAME).write_text('{"corpus_id": "x"}', encoding="utf-8")
        with pytest.raises(SnapshotError, match="not a valid manifest"):
            read_manifest(tmp_path)
