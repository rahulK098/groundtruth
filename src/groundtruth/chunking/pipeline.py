"""Chunking a whole corpus."""

from __future__ import annotations

from collections.abc import Iterable

from groundtruth.chunking.fixed_window import chunk_document, chunking_hash
from groundtruth.chunking.models import Chunk
from groundtruth.chunking.tokenizer import load_tokenizer
from groundtruth.config.models import ChunkingConfig
from groundtruth.corpus.models import Document


def chunk_corpus(documents: Iterable[Document], chunking: ChunkingConfig) -> tuple[Chunk, ...]:
    """Chunk every document with one configuration.

    Documents are processed in ``doc_id`` order so the resulting sequence is
    deterministic regardless of how the corpus was iterated. That determinism
    is what allows the embedding cache to be addressed by content and committed
    (ADR-0003).
    """
    tokenizer = load_tokenizer(chunking.tokenizer_id)
    chunking_id = chunking_hash(chunking)

    chunks: list[Chunk] = []
    for document in sorted(documents, key=lambda doc: doc.doc_id):
        chunks.extend(chunk_document(document, chunking, tokenizer, chunking_id=chunking_id))
    return tuple(chunks)
