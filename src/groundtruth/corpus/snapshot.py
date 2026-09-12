"""Reading and writing corpus snapshots.

A snapshot is a compressed JSONL file plus a manifest, both committed. It is
what makes "clone the repo and reproduce the table" true without a network
fetch.

The load-bearing decision here is that **corpus identity is the sha256 of the
UNCOMPRESSED JSONL bytes**. Compressed bytes depend on the zstd version and
compression level, neither of which is part of what the corpus *is*. Hashing
the compressed form would make an innocuous library upgrade look, to the gate,
exactly like someone swapping the corpus to dodge a regression.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

import zstandard
from pydantic import ValidationError

from groundtruth.corpus.models import CorpusManifest, Document
from groundtruth.corpus.normalize import NORMALIZER_VERSION

SNAPSHOT_FILENAME: Final[str] = "opinions.jsonl.zst"
MANIFEST_FILENAME: Final[str] = "manifest.json"

_COMPRESSION_LEVEL: Final[int] = 10

#: Guard against a corrupt or hostile frame claiming an enormous size.
#: 512 MiB is far beyond this project's ~3 MB snapshot.
_MAX_DECOMPRESSED_BYTES: Final[int] = 512 * 1024 * 1024


class SnapshotError(Exception):
    """A snapshot could not be read, written, or verified."""


def serialize_documents(documents: Iterable[Document]) -> bytes:
    """Render documents to canonical, uncompressed JSONL bytes.

    Documents are sorted by ``doc_id`` and keys are sorted within each line, so
    the same set of documents always produces byte-identical output regardless
    of the order they were fetched in. Without that, re-fetching would change
    the corpus hash and trip the gate for no real reason.
    """
    ordered = sorted(documents, key=lambda doc: doc.doc_id)
    lines = (
        json.dumps(doc.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        for doc in ordered
    )
    return ("\n".join(lines) + "\n").encode("utf-8") if ordered else b""


def content_sha256(documents: Iterable[Document]) -> str:
    """sha256 of the canonical uncompressed serialization."""
    return hashlib.sha256(serialize_documents(documents)).hexdigest()


def write_snapshot(
    documents: Sequence[Document],
    directory: Path,
    *,
    corpus_id: str,
    source: str,
    retrieved_at: str = "",
) -> CorpusManifest:
    """Write a snapshot and its manifest, returning the manifest."""
    directory.mkdir(parents=True, exist_ok=True)

    payload = serialize_documents(documents)
    compressor = zstandard.ZstdCompressor(level=_COMPRESSION_LEVEL)
    (directory / SNAPSHOT_FILENAME).write_bytes(compressor.compress(payload))

    manifest = CorpusManifest(
        corpus_id=corpus_id,
        source=source,
        normalizer_version=NORMALIZER_VERSION,
        doc_count=len(documents),
        content_sha256=hashlib.sha256(payload).hexdigest(),
        retrieved_at=retrieved_at,
    )
    write_manifest(manifest, directory)
    return manifest


def write_manifest(manifest: CorpusManifest, directory: Path) -> None:
    path = directory / MANIFEST_FILENAME
    body = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True)
    path.write_text(body + "\n", encoding="utf-8")


def read_manifest(directory: Path) -> CorpusManifest:
    path = directory / MANIFEST_FILENAME
    if not path.is_file():
        raise SnapshotError(f"corpus manifest not found: {path}")
    try:
        return CorpusManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"could not parse {MANIFEST_FILENAME}: {exc}") from exc
    except ValidationError as exc:
        raise SnapshotError(f"{MANIFEST_FILENAME} is not a valid manifest: {exc}") from exc


def read_snapshot(directory: Path) -> tuple[Document, ...]:
    """Read and decompress a snapshot. Does not verify it -- see `verify_snapshot`."""
    path = directory / SNAPSHOT_FILENAME
    if not path.is_file():
        raise SnapshotError(f"corpus snapshot not found: {path}")

    decompressor = zstandard.ZstdDecompressor()
    try:
        payload = decompressor.decompress(
            path.read_bytes(), max_output_size=_MAX_DECOMPRESSED_BYTES
        )
    except zstandard.ZstdError as exc:
        raise SnapshotError(f"could not decompress {SNAPSHOT_FILENAME}: {exc}") from exc

    documents: list[Document] = []
    for lineno, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            documents.append(Document.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise SnapshotError(
                f"{SNAPSHOT_FILENAME} line {lineno} is not a valid document: {exc}"
            ) from exc
    return tuple(documents)


def verify_snapshot(directory: Path) -> tuple[Document, ...]:
    """Read a snapshot and assert it matches its manifest.

    Checks content hash, document count, and normalizer version. The last one
    catches the subtlest failure: a snapshot written under different
    normalization rules, whose character offsets no longer match the labels
    that index into it.
    """
    manifest = read_manifest(directory)
    documents = read_snapshot(directory)

    actual = content_sha256(documents)
    if actual != manifest.content_sha256:
        raise SnapshotError(
            f"corpus content hash mismatch: manifest says {manifest.content_sha256}, "
            f"snapshot contains {actual}"
        )
    if len(documents) != manifest.doc_count:
        raise SnapshotError(
            f"corpus document count mismatch: manifest says {manifest.doc_count}, "
            f"snapshot contains {len(documents)}"
        )
    if manifest.normalizer_version != NORMALIZER_VERSION:
        raise SnapshotError(
            f"corpus was normalized with NORMALIZER_VERSION "
            f"{manifest.normalizer_version!r} but this code is at "
            f"{NORMALIZER_VERSION!r}; character offsets in the golden set may "
            f"no longer be valid. Re-fetch the corpus and re-validate labels."
        )
    return documents
