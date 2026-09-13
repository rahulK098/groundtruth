"""In-process Okapi BM25.

Roughly a hundred lines, written rather than imported, for two reasons.

First, the project's premise is that every number in the report is understood.
A lexical scorer is the smallest possible piece of that: a term-frequency
saturation term, a length normalization term, and an inverse document
frequency term. Taking `rank_bm25` as a dependency would mean the report says
"BM25" and nobody involved can say which variant.

Second, *which variant* is the whole content of ADR-0007. Postgres'
``ts_rank_cd`` is repeatedly described as full-text ranking and is repeatedly
mistaken for BM25; it has **no IDF term at all**. On a legal corpus IDF is
what separates "summary judgment" from "the court". Having a real BM25 here
is what makes the comparison in that ADR say something.

The scoring function::

    idf(t)   = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

    score(D, Q) = sum over distinct t in Q of
                      idf(t) * tf(t,D) * (k1 + 1)
                      -------------------------------------------
                      tf(t,D) + k1 * (1 - b + b * |D| / avgdl)

The ``ln(1 + x)`` form of IDF is the Lucene variant, chosen over the classic
Robertson-Sparck Jones form deliberately: the classic form goes **negative**
for a term appearing in more than half the corpus, which lets a document be
penalised for containing a query term. On a corpus where "court" appears in
every document, that is not a theoretical concern.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from groundtruth.index.analysis import tokenize
from groundtruth.index.models import ScoredChunk
from groundtruth.retrieval.ordering import take_top_n

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75


class Bm25IndexError(Exception):
    """A BM25 index could not be built or searched."""


def _validate_parameters(k1: float, b: float) -> None:
    if not math.isfinite(k1) or k1 <= 0:
        raise Bm25IndexError(f"k1 must be a positive finite number, got {k1!r}")
    if not math.isfinite(b) or not 0.0 <= b <= 1.0:
        raise Bm25IndexError(f"b ({b!r}) must be a finite number between 0 and 1")


class Bm25Index:
    """Okapi BM25 over an in-memory corpus of chunk texts."""

    backend = "bm25"

    def __init__(
        self,
        chunk_ids: Sequence[str],
        texts: Sequence[str],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        _validate_parameters(k1, b)
        if len(chunk_ids) != len(texts):
            raise Bm25IndexError(
                f"{len(chunk_ids)} chunk ids but {len(texts)} texts; they must align"
            )
        if len(set(chunk_ids)) != len(chunk_ids):
            raise Bm25IndexError("duplicate chunk ids: one chunk must own exactly one row")
        if not chunk_ids:
            # avgdl would be a division by zero, and an index that matches
            # nothing is a far worse outcome than one that refuses to exist.
            raise Bm25IndexError("cannot build a BM25 index over an empty corpus")

        self._k1 = float(k1)
        self._b = float(b)
        # Sorted so the index's internal row layout is a pure function of the
        # corpus rather than of the order it was handed over in. Two indexes
        # built from the same chunks are then byte-identical inside.
        order = sorted(range(len(chunk_ids)), key=lambda i: chunk_ids[i])
        self._chunk_ids: tuple[str, ...] = tuple(str(chunk_ids[i]) for i in order)

        #: term -> (row indices, term frequencies), both as arrays so scoring a
        #: term is one vectorized update rather than a Python loop over
        #: postings. A chunk of pure punctuation simply appears in no posting.
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        raw_postings: dict[str, list[tuple[int, int]]] = {}
        # float64 throughout. The arithmetic is cheap at this scale, and it
        # makes the scores the implementation produces match the formula in
        # the module docstring to full double precision -- which is what lets
        # the known-answer test be a hand computation rather than an
        # approximation with a tolerance chosen to make it pass.
        lengths = np.zeros(len(order), dtype=np.float64)

        for row, source_index in enumerate(order):
            tokens = tokenize(texts[source_index])
            lengths[row] = len(tokens)
            for term, frequency in Counter(tokens).items():
                raw_postings.setdefault(term, []).append((row, frequency))

        for term, postings in raw_postings.items():
            rows = np.fromiter((row for row, _ in postings), dtype=np.int64, count=len(postings))
            frequencies = np.fromiter(
                (freq for _, freq in postings), dtype=np.float64, count=len(postings)
            )
            self._postings[term] = (rows, frequencies)

        self._lengths = lengths
        # Empty chunks are counted in the mean. Excluding them would make
        # avgdl -- and therefore every score in the index -- depend on how many
        # unmatchable chunks the chunker happened to produce.
        average_length = float(lengths.mean())
        self._length_norm = self._k1 * (
            1.0 - self._b + self._b * (lengths / average_length if average_length else lengths)
        )

        document_count = len(self._chunk_ids)
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (document_count - len(rows) + 0.5) / (len(rows) + 0.5))
            for term, (rows, _) in self._postings.items()
        }

    @property
    def count(self) -> int:
        return len(self._chunk_ids)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return self._chunk_ids

    def search(self, query: str, top_n: int) -> tuple[ScoredChunk, ...]:
        """Return up to ``top_n`` matching chunks, deterministically ordered.

        Chunks matching no query term are **not** returned. Returning them
        with a score of zero would hand fusion an arbitrary rank for a
        document carrying no evidence at all.
        """
        # dict.fromkeys, not set: a repeated query term must be counted once
        # (doubling a word is a property of the typist, not of the corpus),
        # and the iteration order must stay fixed so float summation is
        # reproducible.
        terms = [term for term in dict.fromkeys(tokenize(query)) if term in self._postings]
        if not terms:
            return ()

        scores = np.zeros(self.count, dtype=np.float64)
        for term in terms:
            rows, frequencies = self._postings[term]
            contribution = (
                self._idf[term]
                * frequencies
                * (self._k1 + 1.0)
                / (frequencies + self._length_norm[rows])
            )
            scores[rows] += contribution

        matched = np.flatnonzero(scores > 0.0)
        return take_top_n(
            (ScoredChunk(self._chunk_ids[i], float(scores[i])) for i in matched), top_n
        )
