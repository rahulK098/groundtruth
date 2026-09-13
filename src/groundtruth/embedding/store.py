"""On-disk embedding cache.

Committed to the repository, which drives every decision here.

``vectors.f16.npy``
    Memory-mapped on load, so a 10 MB cache costs milliseconds rather than a
    deserialization pass. float16 halves the committed size; vectors are
    upcast to float32 at load so downstream arithmetic is not silently half
    precision. At 384 dimensions the rounding perturbs only near-ties, which
    the deterministic tie-break absorbs.

``keys.jsonl``
    One JSON object per line, row-aligned with the vector matrix. Greppable
    and line-diffable, so git deltas it and a reviewer can see exactly which
    entries changed -- neither of which is true of a binary index.

``meta.json``
    Model identity and the build settings that affect reproducibility.

Rejected alternatives: one file per chunk (tens of thousands of tiny files
ruin git performance and are miserable on Windows); pickle (unsafe to load,
not portable, not diffable); vectors in Postgres only (the gate would then
need a database, defeating ADR-0001).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

VECTORS_FILENAME: Final[str] = "vectors.f16.npy"
KEYS_FILENAME: Final[str] = "keys.jsonl"
META_FILENAME: Final[str] = "meta.json"

#: Bump when the on-disk layout changes in a way that makes older stores
#: unreadable, so the failure is named rather than a confusing crash.
STORE_FORMAT_VERSION: Final[int] = 1


class EmbeddingStoreError(Exception):
    """The embedding cache could not be read, written, or verified."""


@dataclass(frozen=True)
class LoadedStore:
    """An embedding cache loaded into memory."""

    vectors: np.ndarray
    rows: Mapping[str, int]
    meta: Mapping[str, Any]

    @property
    def count(self) -> int:
        return len(self.rows)

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else int(self.meta["dimension"])

    def __contains__(self, key: str) -> bool:
        return key in self.rows

    def vector_for(self, key: str) -> np.ndarray:
        return np.asarray(self.vectors[self.rows[key]])


def store_dir(root: Path, model_id: str, revision: str) -> Path:
    """Directory for one (model, revision) pair.

    The organisation prefix is dropped and the revision truncated: the full
    values live in ``meta.json``, and a 40-character SHA in a path is a
    Windows path-length hazard for no benefit.
    """
    slug = model_id.rsplit("/", maxsplit=1)[-1]
    return root / f"{slug}__{revision[:12]}"


def write_store(
    directory: Path,
    keys: Sequence[str],
    vectors: np.ndarray,
    *,
    model_id: str,
    revision: str,
    normalize: bool,
    build_settings: Mapping[str, Any] | None = None,
) -> None:
    """Write a cache atomically enough to be committed."""
    if len(keys) != vectors.shape[0]:
        raise EmbeddingStoreError(
            f"{len(keys)} keys but {vectors.shape[0]} vector rows; they must align"
        )
    if len(set(keys)) != len(keys):
        raise EmbeddingStoreError(
            "duplicate keys: a key must address exactly one row, or lookups "
            "silently return one of several different embeddings"
        )

    directory.mkdir(parents=True, exist_ok=True)

    np.save(directory / VECTORS_FILENAME, vectors.astype(np.float16))

    lines = [
        json.dumps({"row": row, "key": key}, sort_keys=True, ensure_ascii=False)
        for row, key in enumerate(keys)
    ]
    (directory / KEYS_FILENAME).write_text(
        ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
    )

    meta: dict[str, Any] = {
        "format_version": STORE_FORMAT_VERSION,
        "model_id": model_id,
        "revision": revision,
        "normalize": normalize,
        "dimension": int(vectors.shape[1]),
        "count": len(keys),
        "storage_dtype": "float16",
        "created_at": datetime.now(UTC).isoformat(),
        # Recorded because they affect low-order bits. Regenerating on other
        # hardware may differ slightly; this is what makes that diagnosable
        # rather than mysterious.
        **dict(build_settings or {}),
    }
    (directory / META_FILENAME).write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_store(directory: Path) -> LoadedStore:
    """Load a cache. Does not verify it -- see `verify_store`."""
    vectors_path = directory / VECTORS_FILENAME
    keys_path = directory / KEYS_FILENAME
    meta_path = directory / META_FILENAME

    for path in (vectors_path, keys_path, meta_path):
        if not path.is_file():
            raise EmbeddingStoreError(f"embedding cache not found: {path}")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EmbeddingStoreError(f"could not parse {META_FILENAME}: {exc}") from exc

    # mmap: a committed cache is read on every run and never mutated in place.
    raw = np.load(vectors_path, mmap_mode="r")
    vectors = np.asarray(raw, dtype=np.float32)

    rows: dict[str, int] = {}
    for lineno, line in enumerate(keys_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            rows[str(entry["key"])] = int(entry["row"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise EmbeddingStoreError(f"{KEYS_FILENAME} line {lineno} is malformed: {exc}") from exc

    return LoadedStore(vectors=vectors, rows=rows, meta=meta)


def verify_store(directory: Path) -> int:
    """Assert a cache is internally consistent, returning its entry count.

    The failure this exists to catch is keys and vectors drifting out of
    alignment. That serves the wrong embedding for every row past the break --
    plausible numbers, silently wrong, and invisible to every other check.
    """
    store = read_store(directory)
    expected_rows = int(store.meta.get("count", -1))
    expected_dim = int(store.meta.get("dimension", -1))

    if len(store.rows) != expected_rows:
        raise EmbeddingStoreError(
            f"key count ({len(store.rows)}) does not match meta.json ({expected_rows})"
        )
    if store.vectors.shape[0] != expected_rows:
        raise EmbeddingStoreError(
            f"vector row count ({store.vectors.shape[0]}) does not match "
            f"meta.json ({expected_rows})"
        )
    if expected_rows and store.vectors.shape[1] != expected_dim:
        raise EmbeddingStoreError(
            f"vector dimension ({store.vectors.shape[1]}) does not match meta.json ({expected_dim})"
        )

    max_row = max(store.rows.values(), default=-1)
    if max_row >= store.vectors.shape[0]:
        raise EmbeddingStoreError(
            f"{KEYS_FILENAME} references row {max_row} but only {store.vectors.shape[0]} rows exist"
        )

    return len(store.rows)
