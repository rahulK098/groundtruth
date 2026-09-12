"""Corpus value objects."""

from __future__ import annotations

import hashlib

from pydantic import Field

from groundtruth.config.hashing import canonical_json
from groundtruth.frozen import FrozenModel


class Document(FrozenModel):
    """One normalized corpus document.

    ``text`` is always post-normalization. Golden-set relevance labels are
    character offsets into exactly this string, so a Document carrying raw text
    would silently invalidate every label that points into it.
    """

    doc_id: str = Field(min_length=1)
    text: str
    title: str = ""
    court: str = ""
    date_filed: str = ""
    source_url: str = ""


class CorpusManifest(FrozenModel):
    """Identity of a corpus snapshot.

    The regression gate treats a change to this manifest as a hard failure.
    That is the anti-cheat: without it, a regression could be "fixed" by
    swapping in a corpus the queries happen to do better on.
    """

    corpus_id: str = Field(min_length=1)
    source: str
    #: Which normalization produced the text these offsets index into.
    #: Changing the rules without bumping this would move every label with
    #: nothing to detect it.
    normalizer_version: str = Field(min_length=1)
    doc_count: int = Field(ge=0)
    #: sha256 of the UNCOMPRESSED JSONL bytes -- see snapshot.py for why
    #: identity deliberately ignores the compressed form.
    content_sha256: str = Field(min_length=1)
    retrieved_at: str = ""

    @property
    def manifest_hash(self) -> str:
        """Stable identity of this manifest, as recorded in every result file."""
        payload = self.model_dump(mode="json")
        digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        return f"sha256:{digest}"
