"""Assembling a retriever for the service path (ADR-0001).

Same ``Retriever``, same pipeline, same chunk lookup as the evaluation path;
only the two storage primitives are swapped for their Postgres versions. The
chunk lookup is still built in-process from the committed corpus -- it is
deterministic and cheap -- and ``Retriever`` itself checks at construction
that every chunk id loaded into Postgres resolves in it. A database loaded
under a different chunking therefore fails loudly rather than returning
passages whose spans do not match their text.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from groundtruth.chunking.pipeline import chunk_corpus
from groundtruth.config.models import RetrievalConfig
from groundtruth.corpus.models import Document
from groundtruth.embedding.protocol import Embedder
from groundtruth.index.postgres import PgFtsIndex, PgVectorIndex, index_key
from groundtruth.rerank.protocol import Reranker
from groundtruth.retrieval.pipeline import Retriever

if TYPE_CHECKING:  # pragma: no cover
    from psycopg import Connection


def served_config(config: RetrievalConfig) -> RetrievalConfig:
    """The configuration the service path actually runs.

    The service's lexical arm is ``pg_fts`` (ADR-0001). A hybrid config whose
    file says ``bm25`` is therefore NOT what the service executes -- and a
    result labelled with the file's name and hash would claim BM25 ranked it
    (ADR-0007). So the served variant carries its own backend, its own name
    and therefore its own content hash. Dense configs are served unchanged.
    """
    lexical = config.lexical
    if lexical is None or lexical.backend == "pg_fts":
        return config
    return config.model_copy(
        update={
            "name": f"{config.name}+pg_fts",
            "lexical": lexical.model_copy(update={"backend": "pg_fts"}),
        }
    )


def build_postgres_retriever(
    config: RetrievalConfig,
    documents: Iterable[Document],
    embedder: Embedder,
    conn: Connection[Any],
    *,
    reranker: Reranker | None = None,
) -> Retriever:
    """A retriever over rows already loaded by `gt db load`."""
    served = served_config(config)
    chunks = chunk_corpus(documents, served.chunking)
    key = index_key(served.chunking, served.embedding)

    lexical = None
    if served.lexical is not None:
        lexical = PgFtsIndex(conn, key, served.lexical.language)

    return Retriever(
        served,
        embedder=embedder,
        vector_index=PgVectorIndex(conn, key, served.embedding.dimension),
        lexical_index=lexical,
        chunks={chunk.chunk_id: chunk for chunk in chunks},
        reranker=reranker,
    )
