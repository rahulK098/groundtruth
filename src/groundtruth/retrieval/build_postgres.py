"""Assembling a retriever for the service path (ADR-0001, ADR-0014).

Same ``Retriever``, same pipeline, same chunk lookup as the evaluation path.
Exactly one storage primitive differs: the dense arm reads pgvector instead of
a NumPy matrix. The lexical arm is the same in-process BM25 the gate measures
(ADR-0014) -- serving ``pg_fts`` instead made the hybrid worse than dense-only
-- unless a config explicitly asks for ``pg_fts``, which stays available so
the IDF gap remains measurable.

The chunk lookup is still built in-process from the committed corpus, and
``Retriever`` checks at construction that every chunk id loaded into Postgres
resolves in it. A database loaded under a different chunking therefore fails
loudly rather than returning passages whose spans do not match their text.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from groundtruth.chunking.pipeline import chunk_corpus
from groundtruth.config.models import RetrievalConfig
from groundtruth.corpus.models import Document
from groundtruth.embedding.protocol import Embedder
from groundtruth.index.postgres import PgFtsIndex, PgVectorIndex, index_key
from groundtruth.index.protocol import LexicalIndex
from groundtruth.rerank.protocol import Reranker
from groundtruth.retrieval.build import build_lexical_index
from groundtruth.retrieval.pipeline import Retriever

if TYPE_CHECKING:  # pragma: no cover
    from psycopg import Connection


def build_postgres_retriever(
    config: RetrievalConfig,
    documents: Iterable[Document],
    embedder: Embedder,
    conn: Connection[Any],
    *,
    reranker: Reranker | None = None,
) -> Retriever:
    """A retriever whose dense arm reads rows already loaded by `gt db load`."""
    chunks = chunk_corpus(documents, config.chunking)
    key = index_key(config.chunking, config.embedding)

    lexical: LexicalIndex | None
    if config.lexical is not None and config.lexical.backend == "pg_fts":
        lexical = PgFtsIndex(conn, key, config.lexical.language)
    else:
        lexical = build_lexical_index(config, chunks)

    return Retriever(
        config,
        embedder=embedder,
        vector_index=PgVectorIndex(conn, key, config.embedding.dimension),
        lexical_index=lexical,
        chunks={chunk.chunk_id: chunk for chunk in chunks},
        reranker=reranker,
    )
