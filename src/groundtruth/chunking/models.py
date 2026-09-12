"""Chunk value object."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from groundtruth.frozen import FrozenModel


class Chunk(FrozenModel):
    """One retrievable unit of text.

    Carries **both** token and character spans, and that is the whole point.

    Token spans are how the chunker works. Character spans are how relevance
    labels are expressed (ADR-0002), so scoring can only resolve a labeled span
    to a chunk if every chunk knows exactly which characters of the source
    document it covers.

    The invariant that makes this sound is::

        chunk.text == document.text[chunk.char_start:chunk.char_end]

    It is property-tested against the real tokenizer.
    """

    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    text: str

    #: Character span into the NORMALIZED document text.
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    #: Token span, excluding special tokens.
    token_start: int = Field(ge=0)
    token_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _spans_must_be_well_formed(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) must be greater than char_start ({self.char_start})"
            )
        if self.token_end <= self.token_start:
            raise ValueError(
                f"token_end ({self.token_end}) must be greater than "
                f"token_start ({self.token_start})"
            )
        if len(self.text) != self.char_end - self.char_start:
            # Catches a chunk whose text was built from anything other than the
            # span it claims -- which would silently break span resolution.
            raise ValueError(
                f"text length ({len(self.text)}) does not match the character "
                f"span ({self.char_end - self.char_start})"
            )
        return self

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start
