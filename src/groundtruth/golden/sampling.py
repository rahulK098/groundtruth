"""Choosing passages to propose candidates from.

Two properties matter, and the second is easy to miss.

**Deterministic.** Same corpus, same seed, same passages. Candidate generation
is expensive and is re-run whenever the prompt changes; a sampler that drifted
would make two generation runs incomparable.

**Not aligned to any evaluated chunking.** Passages are character windows at
pseudo-random offsets, deliberately not 512- or 256-token chunks. If the
generator were shown exactly the chunks one configuration produces, the spans
it quotes would tend to sit inside those boundaries -- and the golden set
would quietly favour that chunk size in the comparison it exists to referee.
The window size is a round character count precisely so it corresponds to no
configuration under test.

Passages are drawn round-robin across documents rather than uniformly at
random over the corpus, because opinion lengths vary by an order of magnitude
and uniform sampling would concentrate the golden set in a handful of long
cases.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from groundtruth.corpus.models import Document

#: Long enough to contain a holding with its context, short enough to read in
#: a review pass. Not a token count, and not any configured chunk size.
DEFAULT_WINDOW_CHARS: Final[int] = 1500

#: Fixed so a generation run is reproducible from the corpus alone.
DEFAULT_SEED: Final[int] = 20240101

#: Documents shorter than this carry no holding worth retrieving.
_MIN_DOCUMENT_CHARS: Final[int] = 2000


class SamplingError(Exception):
    """Passages could not be sampled."""


@dataclass(frozen=True, slots=True)
class SampledPassage:
    """One window of one document, with the offsets it came from."""

    doc_id: str
    char_start: int
    char_end: int
    text: str

    @property
    def passage_id(self) -> str:
        return f"{self.doc_id}:{self.char_start}-{self.char_end}"


def _window(document: Document, rng: random.Random, window_chars: int) -> SampledPassage:
    span = min(window_chars, len(document.text))
    start = rng.randrange(0, len(document.text) - span + 1)
    return SampledPassage(
        doc_id=document.doc_id,
        char_start=start,
        char_end=start + span,
        text=document.text[start : start + span],
    )


def sample_passages(
    documents: Sequence[Document],
    *,
    count: int,
    window_chars: int = DEFAULT_WINDOW_CHARS,
    seed: int = DEFAULT_SEED,
) -> tuple[SampledPassage, ...]:
    """Draw ``count`` passages, spread across the corpus and reproducible."""
    if count <= 0:
        raise SamplingError(f"count must be positive, got {count}")
    if window_chars <= 0:
        raise SamplingError(f"window_chars must be positive, got {window_chars}")

    eligible = sorted(
        (doc for doc in documents if len(doc.text) >= _MIN_DOCUMENT_CHARS),
        key=lambda doc: doc.doc_id,
    )
    if not eligible:
        raise SamplingError(
            f"no document reaches {_MIN_DOCUMENT_CHARS} characters; there is "
            f"nothing substantial enough to propose a query about"
        )

    rng = random.Random(seed)
    seen: set[str] = set()
    passages: list[SampledPassage] = []

    # Round-robin, so the first pass covers every document before any document
    # is sampled twice.
    attempts = 0
    max_attempts = count * 20
    while len(passages) < count and attempts < max_attempts:
        document = eligible[len(passages) % len(eligible)]
        passage = _window(document, rng, window_chars)
        attempts += 1
        if passage.passage_id in seen:
            # Overlapping windows on the same document are fine; an identical
            # one is a wasted generation call.
            continue
        seen.add(passage.passage_id)
        passages.append(passage)

    if len(passages) < count:  # pragma: no cover - needs a pathologically small corpus
        raise SamplingError(
            f"could only draw {len(passages)} distinct passages of "
            f"{window_chars} characters from {len(eligible)} documents"
        )

    return tuple(passages)
