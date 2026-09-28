"""The committed rerank-score cache (ADR-0012).

``pairs.jsonl``
    One line per query: ``{query_key, compute_ms, scores: {passage_key: score}}``,
    sorted by ``query_key`` so a rebuild shows a reviewable, per-query diff.

``meta.json``
    Model identity and the build settings that affect the recorded scores and
    latencies -- device, threads, batch size, max length.

Two rerankers read it, and the difference between them is the point, exactly
as with the embedding cache:

``CachedOnlyReranker`` serves recorded scores and **raises on a miss**. The
gate uses it, so the reranker arm can be gated with no model installed.

``RecordingReranker`` wraps a live model, serves what is already recorded,
computes the rest, and hands back a merged store to write. Used only by
``gt cache rerank``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from groundtruth.config.hashing import content_hash
from groundtruth.rerank.protocol import Reranker, RerankScores

PAIRS_FILENAME: Final[str] = "pairs.jsonl"
META_FILENAME: Final[str] = "meta.json"
STORE_FORMAT_VERSION: Final[int] = 1

#: 16 bytes -> 32 hex characters. Keys are never eyeballed, and the passage
#: key space (every chunk of every config) is larger than a config hash's.
_KEY_DIGEST_SIZE: Final[int] = 16

_REGENERATE = (
    "Regenerate and commit it:\n\n"
    "    uv sync --frozen --extra dev --extra models\n"
    "    uv run gt cache rerank\n"
    "    git add data/cache/rerank && git commit -m 'chore: regenerate rerank cache'\n\n"
    "The cache is deliberately authoritative: computing the missing scores on the "
    "fly would need the model in the gate path, which ADR-0003 rules out."
)


class RerankCacheMissError(Exception):
    """The committed rerank cache does not cover a requested (query, passage) pair."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def query_key(query: str, *, model_id: str, revision: str) -> str:
    """Content address of one query under one model.

    Model identity, revision included, is part of the key: upgrading weights
    must never serve scores the old weights produced.
    """
    return content_hash(
        {"model_id": model_id, "revision": revision, "query_sha256": _sha256(query)},
        digest_size=_KEY_DIGEST_SIZE,
    )


def passage_key(passage: str) -> str:
    """Content address of one passage's text.

    Text, not chunk id: chunk ids are a function of the chunking config, and
    the text is what the model actually saw. A chunking change is therefore a
    clean miss rather than a stale hit.
    """
    return content_hash({"text_sha256": _sha256(passage)}, digest_size=_KEY_DIGEST_SIZE)


@dataclass(frozen=True, slots=True)
class QueryRecord:
    """Every score recorded for one query, and what the model took to produce them."""

    compute_ms: float
    scores: Mapping[str, float]


@dataclass(frozen=True)
class RerankStore:
    model_id: str
    revision: str
    queries: Mapping[str, QueryRecord]
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def pair_count(self) -> int:
        return sum(len(record.scores) for record in self.queries.values())


def rerank_store_dir(root: Path, model_id: str, revision: str) -> Path:
    """Directory for one (model, revision) pair, named like the embedding store."""
    slug = model_id.rsplit("/", maxsplit=1)[-1]
    return root / f"{slug}__{revision[:12]}"


def write_rerank_store(
    directory: Path, store: RerankStore, *, build_settings: Mapping[str, Any] | None = None
) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    lines = [
        json.dumps(
            {
                "query_key": key,
                "compute_ms": record.compute_ms,
                "scores": dict(sorted(record.scores.items())),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        for key, record in sorted(store.queries.items())
    ]
    (directory / PAIRS_FILENAME).write_text(
        ("\n".join(lines) + "\n") if lines else "", encoding="utf-8", newline="\n"
    )

    meta: dict[str, Any] = {
        "format_version": STORE_FORMAT_VERSION,
        "model_id": store.model_id,
        "revision": store.revision,
        "queries": len(store.queries),
        "pairs": store.pair_count,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **dict(build_settings or {}),
    }
    (directory / META_FILENAME).write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def read_rerank_store(directory: Path) -> RerankStore:
    pairs_path = directory / PAIRS_FILENAME
    meta_path = directory / META_FILENAME
    if not pairs_path.is_file() or not meta_path.is_file():
        raise RerankCacheMissError(f"no rerank cache at {directory}.\n\n{_REGENERATE}")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RerankCacheMissError(f"{directory.name}/{META_FILENAME} is malformed: {exc}") from exc

    queries: dict[str, QueryRecord] = {}
    for lineno, line in enumerate(pairs_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            queries[str(entry["query_key"])] = QueryRecord(
                compute_ms=float(entry["compute_ms"]),
                scores={str(k): float(v) for k, v in entry["scores"].items()},
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RerankCacheMissError(
                f"{directory.name}/{PAIRS_FILENAME} line {lineno} is malformed: {exc}"
            ) from exc

    return RerankStore(
        model_id=str(meta["model_id"]),
        revision=str(meta["revision"]),
        queries=queries,
        meta=meta,
    )


class CachedOnlyReranker:
    """Serves committed scores and refuses to compute anything."""

    def __init__(self, store: RerankStore) -> None:
        self._store = store

    @property
    def model_id(self) -> str:
        return self._store.model_id

    @property
    def revision(self) -> str:
        return self._store.revision

    def score(self, query: str, passages: Sequence[str]) -> RerankScores:
        if not passages:
            return RerankScores(scores=(), compute_ms=0.0)

        keys = [passage_key(passage) for passage in passages]
        record = self._store.queries.get(
            query_key(query, model_id=self.model_id, revision=self.revision)
        )
        missing = len(keys) if record is None else sum(1 for k in keys if k not in record.scores)
        if record is None or missing:
            raise RerankCacheMissError(
                f"{missing} of {len(keys)} (query, passage) pairs are not in the committed "
                f"rerank cache for {self.model_id}@{self.revision[:12]}.\n\n"
                f"This usually means a golden query, a chunking config, or the fused "
                f"candidate set changed without regenerating the cache. {_REGENERATE}"
            )

        return RerankScores(
            scores=tuple(record.scores[k] for k in keys), compute_ms=record.compute_ms
        )


class RecordingReranker:
    """A live reranker that reuses recorded scores and records new ones."""

    def __init__(self, delegate: Reranker, existing: RerankStore | None) -> None:
        self._delegate = delegate
        if existing is not None and (
            existing.model_id != delegate.model_id or existing.revision != delegate.revision
        ):
            raise ValueError(
                f"existing rerank store is for {existing.model_id}@{existing.revision[:12]} "
                f"but the model is {delegate.model_id}@{delegate.revision[:12]}"
            )
        self._records: dict[str, QueryRecord] = dict(existing.queries) if existing else {}
        self.computed_queries = 0
        self.reused_queries = 0

    @property
    def model_id(self) -> str:
        return self._delegate.model_id

    @property
    def revision(self) -> str:
        return self._delegate.revision

    def score(self, query: str, passages: Sequence[str]) -> RerankScores:
        qk = query_key(query, model_id=self.model_id, revision=self.revision)
        keys = [passage_key(passage) for passage in passages]
        record = self._records.get(qk)

        if record is not None and all(k in record.scores for k in keys):
            self.reused_queries += 1
            return RerankScores(
                scores=tuple(record.scores[k] for k in keys), compute_ms=record.compute_ms
            )

        # Recompute the whole batch, not just the missing pairs: the recorded
        # compute_ms must be the cost of a real candidate batch.
        result = self._delegate.score(query, passages)
        if len(result.scores) != len(passages):
            raise ValueError(
                f"reranker returned {len(result.scores)} scores for {len(passages)} passages"
            )

        merged = dict(record.scores) if record is not None else {}
        merged.update(zip(keys, result.scores, strict=True))
        self._records[qk] = QueryRecord(compute_ms=result.compute_ms, scores=merged)
        self.computed_queries += 1
        return result

    def store(self) -> RerankStore:
        return RerankStore(
            model_id=self.model_id, revision=self.revision, queries=dict(self._records)
        )


def load_cached_only_reranker(model_id: str, revision: str, cache_root: Path) -> CachedOnlyReranker:
    """The one line every evaluation path needs for a reranker-enabled config."""
    store = read_rerank_store(rerank_store_dir(cache_root, model_id, revision))
    if store.model_id != model_id or store.revision != revision:
        raise RerankCacheMissError(
            f"rerank cache is for {store.model_id}@{store.revision[:12]}, "
            f"not {model_id}@{revision[:12]}.\n\n{_REGENERATE}"
        )
    return CachedOnlyReranker(store)
