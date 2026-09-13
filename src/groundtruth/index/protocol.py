"""The two seams between the evaluation path and the service path.

These are the *only* interfaces that differ between them (ADR-0001). Keeping
them this narrow is what makes the parity claim checkable: if the Postgres
implementations satisfy these signatures and the shared pipeline above is
untouched, a difference in results has exactly one place to come from.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from groundtruth.index.models import ScoredChunk


@runtime_checkable
class VectorIndex(Protocol):
    """Dense candidate generation."""

    @property
    def count(self) -> int:
        """Number of indexed chunks."""
        ...

    @property
    def dimension(self) -> int:
        """Width of the vectors this index holds."""
        ...

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """Every chunk this index can return, ascending.

        Exposed so a retriever can confirm at construction that every possible
        candidate resolves to a chunk. Discovering that mid-query would make
        the failure depend on which queries happened to surface the gap.
        """
        ...

    def search(self, query_vector: np.ndarray, top_n: int) -> tuple[ScoredChunk, ...]:
        """Return the ``top_n`` nearest chunks, already deterministically ordered."""
        ...


@runtime_checkable
class LexicalIndex(Protocol):
    """Lexical candidate generation.

    Takes the query *string*, not a vector: a lexical backend does its own
    analysis, and the analyzer is part of what distinguishes one backend from
    another (ADR-0007).
    """

    @property
    def count(self) -> int:
        """Number of indexed chunks."""
        ...

    @property
    def backend(self) -> str:
        """Name recorded in results. ``bm25`` and ``pg_fts`` are not the same thing."""
        ...

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """Every chunk this index can return, ascending."""
        ...

    def search(self, query: str, top_n: int) -> tuple[ScoredChunk, ...]:
        """Return up to ``top_n`` matching chunks, already deterministically ordered."""
        ...
