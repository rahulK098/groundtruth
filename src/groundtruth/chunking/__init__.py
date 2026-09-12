"""Chunking: fixed token windows over normalized documents."""

from __future__ import annotations

from groundtruth.chunking.fixed_window import chunk_document, chunking_hash
from groundtruth.chunking.models import Chunk
from groundtruth.chunking.pipeline import chunk_corpus
from groundtruth.chunking.tokenizer import (
    TokenizerNotVendoredError,
    load_tokenizer,
    tokenizer_path,
    tokenizer_slug,
)

__all__ = [
    "Chunk",
    "TokenizerNotVendoredError",
    "chunk_corpus",
    "chunk_document",
    "chunking_hash",
    "load_tokenizer",
    "tokenizer_path",
    "tokenizer_slug",
]
