"""The service path's two storage primitives: pgvector and Postgres full-text.

These are the ONLY things that differ from the evaluation path (ADR-0001).
Both satisfy the same synchronous protocols as ``NumpyVectorIndex`` and
``Bm25Index``, so the shared pipeline above them is byte-for-byte the code
the gate measures -- which is what the parity test checks.

Two naming disciplines carried over from the ADRs:

- ``PgVectorIndex`` builds **no ANN index**. At ~4k rows per index an exact
  scan costs milliseconds; HNSW would add non-determinism for nothing
  (ADR-0001).
- ``PgFtsIndex`` reports its backend as ``pg_fts`` and is never called BM25:
  ``ts_rank_cd`` has no IDF term (ADR-0007).

psycopg is imported lazily. It lives in the ``api`` extra, which the gate
environment never installs, and this module must stay importable there.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from groundtruth.chunking.models import Chunk
from groundtruth.config.hashing import content_hash
from groundtruth.config.models import ChunkingConfig, EmbeddingConfig
from groundtruth.index.models import ScoredChunk

if TYPE_CHECKING:  # pragma: no cover
    from psycopg import Connection

TABLE: Final[str] = "chunks"

#: The one text-search configuration with a precomputed, GIN-indexed column.
_STORED_LANGUAGE: Final[str] = "english"
_STORED_TSV: Final[str] = "tsv_english"


class PostgresIndexError(Exception):
    """A Postgres-backed index could not be loaded, opened or searched."""


class ApiExtraNotInstalledError(Exception):
    """The optional `api` extra (psycopg, pgvector) is not installed."""


def index_key(chunking: ChunkingConfig, embedding: EmbeddingConfig) -> str:
    """Identity of one loaded index: which chunks, embedded by which model.

    Rows for different chunkings (512 vs 256) and models share one table,
    partitioned by this key -- so a config change is a different key, never
    a silent reuse of rows built under different settings.
    """
    return content_hash(
        {
            "chunking": chunking.model_dump(mode="json"),
            "embedding": {
                "model_id": embedding.model_id,
                "revision": embedding.revision,
                "normalize": embedding.normalize,
            },
        }
    )


def connect(url: str) -> Connection[Any]:
    """Open a connection with the pgvector type adapter registered."""
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ApiExtraNotInstalledError(
            "The 'api' extra is not installed, so the Postgres path is unavailable. "
            "This is expected in the gate environment.\n\n"
            "    uv sync --frozen --extra dev --extra api"
        ) from exc

    try:
        conn = psycopg.connect(url, autocommit=True)
    except psycopg.OperationalError as exc:
        raise PostgresIndexError(
            f"could not connect to Postgres ({exc}). Is it running? `docker compose up -d db`"
        ) from exc
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def ensure_schema(conn: Connection[Any], dimension: int) -> None:
    """Create the chunks table if absent. Idempotent.

    Owned by code rather than an initdb script, so there is no "initdb only
    runs on an empty volume" trap: `gt db load` always converges.
    """
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            index_key   text    NOT NULL,
            chunk_id    text    NOT NULL,
            doc_id      text    NOT NULL,
            text        text    NOT NULL,
            char_start  integer NOT NULL,
            char_end    integer NOT NULL,
            embedding   vector({dimension}) NOT NULL,
            PRIMARY KEY (index_key, chunk_id)
        )
        """
    )
    # The tsvector is STORED, not computed per query. Computing
    # to_tsvector over every row at query time cost ~2.2 s per search; a
    # stored column with a GIN index returns the identical ranking in
    # milliseconds. `english` only, because that is the one analyzer any
    # shipped config asks for -- other languages fall back to per-query
    # analysis in PgFtsIndex rather than being silently unsupported.
    conn.execute(
        f"""
        ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS {_STORED_TSV}
            tsvector GENERATED ALWAYS AS (to_tsvector('{_STORED_LANGUAGE}', text)) STORED
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {TABLE}_{_STORED_TSV}_gin ON {TABLE} USING gin ({_STORED_TSV})"
    )


def _normalized(vectors: np.ndarray) -> np.ndarray:
    """Exactly what ``NumpyVectorIndex`` stores: float32, divided by the true norm.

    Storing the same unit vectors is half of what makes parity achievable;
    scoring by inner product over them (see ``PgVectorIndex.search``) is the
    other half.
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    if not norms.all():
        raise PostgresIndexError("a chunk has a zero vector; cosine similarity is undefined")
    return np.asarray(matrix / norms[:, None], dtype=np.float32)


def load_index(
    conn: Connection[Any], key: str, chunks: Sequence[Chunk], vectors: np.ndarray
) -> int:
    """Replace every row under ``key`` with these chunks. Returns the row count.

    Delete-then-insert inside one transaction, so a reader never sees a
    half-loaded index and a reload after a chunking change never leaves
    stale rows behind under the same key.
    """
    if len(chunks) != vectors.shape[0]:
        raise PostgresIndexError(f"{len(chunks)} chunks but {vectors.shape[0]} vectors")

    unit = _normalized(vectors)
    with conn.transaction(), conn.cursor() as cursor:
        cursor.execute(f"DELETE FROM {TABLE} WHERE index_key = %s", (key,))
        with cursor.copy(
            f"COPY {TABLE} (index_key, chunk_id, doc_id, text, char_start, char_end, embedding) "
            f"FROM STDIN"
        ) as copy:
            for chunk, vector in zip(chunks, unit, strict=True):
                copy.write_row(
                    (
                        key,
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.text,
                        chunk.char_start,
                        chunk.char_end,
                        vector,
                    )
                )
    return len(chunks)


def _chunk_ids(conn: Connection[Any], key: str) -> tuple[str, ...]:
    rows = conn.execute(
        f"SELECT chunk_id FROM {TABLE} WHERE index_key = %s ORDER BY chunk_id", (key,)
    ).fetchall()
    if not rows:
        raise PostgresIndexError(
            f"no rows loaded for index {key}. Load them first: `uv run gt db load`"
        )
    return tuple(str(row[0]) for row in rows)


class PgVectorIndex:
    """Exact inner-product search over unit vectors in pgvector. No ANN index."""

    backend = "pgvector"

    def __init__(self, conn: Connection[Any], key: str, dimension: int) -> None:
        self._conn = conn
        self._key = key
        self._dimension = dimension
        self._chunk_ids = _chunk_ids(conn, key)

    @property
    def count(self) -> int:
        return len(self._chunk_ids)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return self._chunk_ids

    def search(self, query_vector: np.ndarray, top_n: int) -> tuple[ScoredChunk, ...]:
        if top_n <= 0:
            raise PostgresIndexError(f"top_n must be positive, got {top_n}")

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self._dimension:
            raise PostgresIndexError(
                f"query has dimension {query.shape[0]} but the index holds {self._dimension}"
            )
        norm = float(np.linalg.norm(query))
        if norm == 0.0 or not np.isfinite(query).all():
            raise PostgresIndexError("query vector is zero or non-finite")
        unit = np.asarray(query / norm, dtype=np.float32)

        # Inner product over unit vectors IS cosine similarity. `<#>` is
        # pgvector's NEGATIVE inner product, hence the sign flip. The
        # `chunk_id` tie-break is the project's one ordering rule (ADR-0001);
        # Postgres applies it here so the rows arrive already in total order.
        rows = self._conn.execute(
            f"""
            SELECT chunk_id, -(embedding <#> %s) AS score
            FROM {TABLE}
            WHERE index_key = %s
            ORDER BY score DESC, chunk_id ASC
            LIMIT %s
            """,
            (unit, self._key, top_n),
        ).fetchall()
        return tuple(ScoredChunk(str(chunk_id), float(score)) for chunk_id, score in rows)


class PgFtsIndex:
    """Postgres full-text search, ranked by ``ts_rank_cd``. NOT BM25 (ADR-0007)."""

    backend = "pg_fts"

    def __init__(self, conn: Connection[Any], key: str, language: str = "english") -> None:
        self._conn = conn
        self._key = key
        self._language = language
        self._chunk_ids = _chunk_ids(conn, key)

    @property
    def count(self) -> int:
        return len(self._chunk_ids)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return self._chunk_ids

    def search(self, query: str, top_n: int) -> tuple[ScoredChunk, ...]:
        if top_n <= 0:
            raise PostgresIndexError(f"top_n must be positive, got {top_n}")

        # plainto_tsquery ANDs every term, which would make almost any
        # natural-language question match nothing. Rewritten to OR so a chunk
        # matching ANY term is a candidate -- the same candidate rule as the
        # in-process BM25 arm, which is what keeps the two lexical arms
        # comparable in everything except their ranking functions.
        # The stored, GIN-indexed column when the language has one; the same
        # expression computed per row otherwise. Both yield the same
        # tsvector, so the choice changes latency, never the ranking.
        document = (
            _STORED_TSV
            if self._language == _STORED_LANGUAGE
            else "to_tsvector(%(cfg)s::regconfig, text)"
        )
        rows = self._conn.execute(
            f"""
            WITH q AS (
                SELECT replace(plainto_tsquery(%(cfg)s::regconfig, %(query)s)::text,
                               '&', '|')::tsquery AS query
            )
            SELECT chunk_id, ts_rank_cd({document}, q.query) AS score
            FROM {TABLE}, q
            WHERE index_key = %(key)s
              AND q.query::text <> ''
              AND {document} @@ q.query
            ORDER BY score DESC, chunk_id ASC
            LIMIT %(n)s
            """,
            {"cfg": self._language, "query": query, "key": self._key, "n": top_n},
        ).fetchall()
        return tuple(ScoredChunk(str(chunk_id), float(score)) for chunk_id, score in rows)


def loaded_keys(conn: Connection[Any]) -> dict[str, int]:
    """Row count per loaded index -- what `/readyz` and `gt db status` report.

    An empty mapping, not an error, before the first `gt db load`: "nothing
    loaded yet" is a state to report, not a failure to raise.
    """
    exists = conn.execute("SELECT to_regclass(%s) IS NOT NULL", (TABLE,)).fetchone()
    if not exists or not exists[0]:
        return {}
    rows = conn.execute(f"SELECT index_key, count(*) FROM {TABLE} GROUP BY index_key").fetchall()
    return {str(key): int(count) for key, count in rows}
