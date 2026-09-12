"""Corpus ingestion: normalization, snapshotting, and source fetching."""

from __future__ import annotations

from groundtruth.corpus.models import CorpusManifest, Document
from groundtruth.corpus.normalize import NORMALIZER_VERSION, normalize_text
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

__all__ = [
    "MANIFEST_FILENAME",
    "NORMALIZER_VERSION",
    "SNAPSHOT_FILENAME",
    "CorpusManifest",
    "Document",
    "SnapshotError",
    "content_sha256",
    "normalize_text",
    "read_manifest",
    "read_snapshot",
    "serialize_documents",
    "verify_snapshot",
    "write_manifest",
    "write_snapshot",
]
