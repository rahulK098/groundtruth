"""Fixed-size token-window chunking.

Windows are measured in **tokens**, using the model's own tokenizer, so
``chunk_size`` means what the encoder thinks it means rather than an
approximation in characters or words. A 512-character window and a 512-token
window differ by roughly 4x, and the whole point of the chunk-size arm is that
the number being compared is meaningful.

Every chunk records the character span it covers, because relevance labels are
character spans (ADR-0002) and scoring resolves them against whatever chunks a
configuration produced.
"""

from __future__ import annotations

from tokenizers import Tokenizer

from groundtruth.chunking.models import Chunk
from groundtruth.config.hashing import content_hash
from groundtruth.config.models import ChunkingConfig
from groundtruth.corpus.models import Document


def chunking_hash(chunking: ChunkingConfig) -> str:
    """Content address of a chunking configuration.

    Part of every chunk id, so chunks produced by different chunking settings
    can never collide -- which matters because the comparison runs several
    chunk sizes over the same corpus at once.
    """
    return content_hash(chunking.model_dump(mode="json"))


def _chunk_id(chunking_id: str, doc_id: str, token_start: int, token_end: int) -> str:
    return content_hash(
        {
            "chunking": chunking_id,
            "doc_id": doc_id,
            "token_start": token_start,
            "token_end": token_end,
        }
    )


def chunk_document(
    document: Document,
    chunking: ChunkingConfig,
    tokenizer: Tokenizer,
    *,
    chunking_id: str | None = None,
) -> tuple[Chunk, ...]:
    """Split one document into overlapping fixed-size token windows.

    ``chunking_id`` may be supplied to avoid recomputing it per document when
    chunking a whole corpus.
    """
    chunking_id = chunking_id or chunking_hash(chunking)

    # add_special_tokens=False: [CLS]/[SEP] carry zero-width offsets and are an
    # artifact of the encoder's input format, not content. Including them would
    # consume window budget and corrupt the character span of every chunk.
    encoding = tokenizer.encode(document.text, add_special_tokens=False)
    offsets: list[tuple[int, int]] = list(encoding.offsets)
    token_count = len(offsets)
    if token_count == 0:
        return ()

    stride = chunking.chunk_size - chunking.chunk_overlap  # validated > 0
    chunks: list[Chunk] = []

    for token_start in range(0, token_count, stride):
        token_end = min(token_start + chunking.chunk_size, token_count)

        char_start = offsets[token_start][0]
        char_end = offsets[token_end - 1][1]

        # Degenerate windows can occur if a tokenizer emits zero-width offsets
        # for a run of tokens. Skipping is correct: an empty chunk would fail
        # Chunk's own span validation and could never match a label anyway.
        if char_end <= char_start:
            if token_end == token_count:
                break
            continue

        chunks.append(
            Chunk(
                chunk_id=_chunk_id(chunking_id, document.doc_id, token_start, token_end),
                doc_id=document.doc_id,
                text=document.text[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
                token_start=token_start,
                token_end=token_end,
            )
        )

        # The final window is already covered; stepping again would emit a
        # short tail chunk wholly contained in the one just produced.
        if token_end == token_count:
            break

    return tuple(chunks)
