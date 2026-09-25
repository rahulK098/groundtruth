"""Recall@k, MRR@k and nDCG@k -- span-level, span-deduplicated (ADR-0004).

Each function walks the ranked list once, following its formula in
`../../../docs/methodology.md` almost line for line, rather than sharing one
clever incremental computation across metrics. At the scale this project
runs at (k <= 10, 1-3 labeled spans per query) the cost difference is
nothing; the readability difference is the entire reason ADR-0004 rejects
importing a scorer library.

Every function returns ``None`` for a query with no ``gain >= 2`` span --
"skip", never a silent 0.0. ``GoldenPair`` already forbids that case, so in
practice this branch is defensive rather than reachable, and the tests reach
it directly with a raw label list to prove it works anyway (the methodology
says "asserted anyway" for exactly this reason).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from groundtruth.golden.models import MIN_RELEVANT_GAIN, RelevanceLabel
from groundtruth.retrieval.models import RetrievedPassage
from groundtruth.scoring.span_resolution import chunk_covers_span


def _relevant_spans(labels: Sequence[RelevanceLabel]) -> list[RelevanceLabel]:
    return [label for label in labels if label.gain >= MIN_RELEVANT_GAIN]


def recall_at_k(
    passages: Sequence[RetrievedPassage], labels: Sequence[RelevanceLabel], k: int
) -> float | None:
    """Fraction of gain>=2 spans covered by any chunk in the top-k.

    Span level, not chunk level: one span typically maps to 2-3 overlapping
    chunks, and counting per chunk would penalize a config for the chunker's
    own overlap setting rather than for retrieval quality.
    """
    relevant = _relevant_spans(labels)
    if not relevant:
        return None

    window = passages[:k]
    covered = sum(1 for span in relevant if any(chunk_covers_span(p, span) for p in window))
    return covered / len(relevant)


def mrr_at_k(
    passages: Sequence[RetrievedPassage], labels: Sequence[RelevanceLabel], k: int
) -> float | None:
    """1 / rank of the first chunk covering any gain>=2 span, else 0.

    Sees only the first relevant span by construction -- Recall and nDCG are
    reported alongside it for that reason (ADR-0004).
    """
    relevant = _relevant_spans(labels)
    if not relevant:
        return None

    for rank, passage in enumerate(passages[:k], start=1):
        if any(chunk_covers_span(passage, span) for span in relevant):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    passages: Sequence[RetrievedPassage], labels: Sequence[RelevanceLabel], k: int
) -> float | None:
    """Graded, span-deduplicated nDCG using the full 0-3 gain scale.

    Walking the ranked list in order, the first chunk covering a span earns
    that span's gain; later chunks covering only already-credited spans earn
    0. A chunk covering several uncredited spans at once earns the max of
    their gains and credits all of them.

    Without this de-duplication, IDCG would depend on how many chunks happen
    to overlap a span -- which varies with chunk_size, silently making nDCG
    incomparable across exactly the configs this project compares.
    """
    if not _relevant_spans(labels):
        return None

    window = passages[:k]
    credited: set[int] = set()
    dcg = 0.0
    for rank, passage in enumerate(window, start=1):
        newly_covered = [
            index
            for index, span in enumerate(labels)
            if index not in credited and chunk_covers_span(passage, span)
        ]
        if not newly_covered:
            continue
        gain_effective = max(labels[index].gain for index in newly_covered)
        dcg += (2**gain_effective - 1) / math.log2(rank + 1)
        credited.update(newly_covered)

    gains_desc = sorted((label.gain for label in labels), reverse=True)[:k]
    idcg = sum(
        (2**gain - 1) / math.log2(position + 1) for position, gain in enumerate(gains_desc, start=1)
    )

    return dcg / idcg if idcg > 0 else 0.0
