"""Query-passage echo detection.

ADR-0009 names the failure this exists to catch: LLM-authored queries tend to
echo their source passage's vocabulary, which inflates lexical *and* dense
retrieval alike and makes every configuration look good. A golden set built
from echoes measures whether retrieval can find text it was handed, which is
not a question anyone needs answered.

The review CLI shows both signals beside each candidate so the reviewer can
see echo rather than having to notice it.

Two numbers, because they catch different things:

``token_overlap``
    Share of the query's distinct tokens that appear anywhere in the passage.
    Catches a query assembled from the passage's vocabulary in a new order.

``longest_common_run``
    Longest run of consecutive tokens the two share. Catches verbatim lifting,
    which unigram overlap alone can miss -- a query can overlap 100% with a
    long passage and still be a genuine paraphrase, whereas an eight-token
    shared run is not a paraphrase of anything.

Neither is a threshold. They are shown to a human, who decides. Auto-rejecting
above some cutoff would be one more untuned knob quietly shaping the data.
"""

from __future__ import annotations

from dataclasses import dataclass

from groundtruth.index.analysis import tokenize


@dataclass(frozen=True, slots=True)
class EchoSignal:
    """How much a query looks like it was copied from its passage."""

    token_overlap: float
    longest_common_run: int
    query_tokens: int


def _longest_common_run(query: tuple[str, ...], passage: tuple[str, ...]) -> int:
    """Length of the longest contiguous token sequence present in both.

    Straightforward dynamic programming over one rolling row: the passage is a
    chunk-sized piece of text and the query is a sentence, so the quadratic
    term is small and the clarity is worth more than the constant factor.
    """
    if not query or not passage:
        return 0

    previous = [0] * (len(passage) + 1)
    best = 0
    for token in query:
        current = [0] * (len(passage) + 1)
        for j, other in enumerate(passage, start=1):
            if token == other:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def echo_signal(query: str, passage: str) -> EchoSignal:
    """Measure how far a query reuses its source passage's wording.

    Uses the same analyzer as the BM25 index, so "shared token" means here
    exactly what it means to the lexical arm.
    """
    query_tokens = tokenize(query)
    passage_tokens = tokenize(passage)

    distinct = dict.fromkeys(query_tokens)
    if not distinct:
        # A query with no word characters at all. Zero rather than a division
        # by zero, and the reviewer will reject it on sight anyway.
        return EchoSignal(token_overlap=0.0, longest_common_run=0, query_tokens=0)

    present = set(passage_tokens)
    shared = sum(1 for token in distinct if token in present)

    return EchoSignal(
        token_overlap=shared / len(distinct),
        longest_common_run=_longest_common_run(query_tokens, passage_tokens),
        query_tokens=len(distinct),
    )
