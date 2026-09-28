"""The retriever: one shared code path over two sets of storage primitives.

Everything here -- query prefixing, candidate pooling, fusion, truncation,
tie-breaking, result assembly -- is identical for the evaluation path the gate
measures and the service path FastAPI serves. Only the two indexes handed in
differ (ADR-0001), and that is what licenses the claim that they are the same
system rather than two implementations that happen to agree.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from time import perf_counter

from groundtruth.chunking.models import Chunk
from groundtruth.config.models import RetrievalConfig
from groundtruth.embedding.protocol import Embedder
from groundtruth.index.models import ScoredChunk
from groundtruth.index.protocol import LexicalIndex, VectorIndex
from groundtruth.rerank.protocol import Reranker
from groundtruth.retrieval.fusion import reciprocal_rank_fusion
from groundtruth.retrieval.models import (
    PassageScores,
    RetrievalResult,
    RetrievedPassage,
    StageLatenciesMs,
)
from groundtruth.retrieval.ordering import order_candidates


class RetrieverError(Exception):
    """The retriever could not be assembled, or could not answer a query."""


@contextmanager
def _timed(into: dict[str, float], stage: str) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        into[stage] = (perf_counter() - start) * 1000.0


class Retriever:
    """Answers queries under one configuration."""

    def __init__(
        self,
        config: RetrievalConfig,
        *,
        embedder: Embedder,
        vector_index: VectorIndex,
        chunks: Mapping[str, Chunk],
        lexical_index: LexicalIndex | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._config = config
        self.embedder = embedder
        self.vector_index = vector_index
        self.lexical_index = lexical_index
        self.reranker = reranker
        self._chunks = chunks

        self._check_mode(config, lexical_index)
        self._check_dimensions(config, embedder, vector_index)
        self._check_chunk_lookup(vector_index, chunks)
        self._check_reranker(config, reranker)

    @staticmethod
    def _check_reranker(config: RetrievalConfig, reranker: Reranker | None) -> None:
        settings = config.reranker
        if not settings.enabled:
            if reranker is not None:
                raise RetrieverError(
                    f"config {config.name!r} disables the reranker, so one supplied "
                    f"here is never called; that is dead wiring"
                )
            return

        if reranker is None:
            # Better a refusal than a result file labelled "..._rerank" that
            # was produced without the stage its name promises.
            raise RetrieverError(
                f"config {config.name!r} enables the reranker but no reranker was supplied"
            )
        if (reranker.model_id, reranker.revision) != (settings.model_id, settings.revision):
            raise RetrieverError(
                f"reranker {reranker.model_id}@{reranker.revision} does not match the "
                f"{settings.model_id}@{settings.revision} declared by config {config.name!r}"
            )

    @staticmethod
    def _check_mode(config: RetrievalConfig, lexical_index: LexicalIndex | None) -> None:
        if config.retrieval_mode == "hybrid" and lexical_index is None:
            raise RetrieverError(
                f"config {config.name!r} is hybrid but no lexical index was supplied"
            )
        if config.retrieval_mode == "dense" and lexical_index is not None:
            # Dead wiring: the pipeline would never call it, and someone would
            # eventually tune the lexical arm and conclude it does nothing.
            raise RetrieverError(
                f"config {config.name!r} is dense, so a lexical index is never "
                f"read; supplying one is dead wiring"
            )

    @staticmethod
    def _check_dimensions(
        config: RetrievalConfig, embedder: Embedder, vector_index: VectorIndex
    ) -> None:
        declared = config.embedding.dimension
        if embedder.dimension != declared:
            raise RetrieverError(
                f"embedder dimension {embedder.dimension} does not match the "
                f"{declared} declared by config {config.name!r}"
            )
        if vector_index.dimension != declared:
            raise RetrieverError(
                f"index dimension {vector_index.dimension} does not match the "
                f"{declared} declared by config {config.name!r}"
            )

    @staticmethod
    def _check_chunk_lookup(vector_index: VectorIndex, chunks: Mapping[str, Chunk]) -> None:
        # Checked once here rather than discovered mid-query. A candidate that
        # cannot be resolved to a chunk has no text and no character span, so
        # it is unscoreable -- and finding that out per query would make the
        # failure depend on which queries happen to surface it.
        missing = [chunk_id for chunk_id in vector_index.chunk_ids if chunk_id not in chunks]
        if missing:
            raise RetrieverError(
                f"{len(missing)} indexed chunks are not in the chunk lookup "
                f"(first: {missing[0]!r}); every candidate must resolve to its "
                f"text and character span"
            )

    @property
    def config(self) -> RetrievalConfig:
        return self._config

    def retrieve(self, query: str, *, top_k: int | None = None) -> RetrievalResult:
        """Rank passages for one query."""
        if not query.strip():
            raise RetrieverError("query is empty; there is nothing to retrieve")

        limit = self._config.top_k if top_k is None else top_k
        timings: dict[str, float] = {}
        started = perf_counter()

        with _timed(timings, "embed"):
            query_vector = self.embedder.embed([self._config.embedding.query_prefix + query])

        with _timed(timings, "dense"):
            dense = self.vector_index.search(query_vector, self._config.dense_top_n)

        lexical, ordered = self._lexical_and_fused(query, dense, timings)
        fused_scores = {candidate.chunk_id: candidate.score for candidate in ordered}

        rerank_scores: dict[str, float] = {}
        # What the reranker says the stage cost, minus what calling it
        # actually took. Zero for a live model; for a cached one it adds back
        # the recorded compute time the lookup stood in for (ADR-0012), so
        # `total` still includes every stage's cost.
        recorded_extra_ms = 0.0
        if self.reranker is not None:
            ordered, rerank_scores, recorded_extra_ms = self._rerank(query, ordered, timings)

        total_ms = (perf_counter() - started) * 1000.0 + recorded_extra_ms

        return RetrievalResult(
            query=query,
            config_name=self._config.name,
            config_hash=self._config.config_hash,
            passages=self._assemble(ordered[:limit], dense, lexical, fused_scores, rerank_scores),
            latency_ms=StageLatenciesMs(total=total_ms, **timings),
        )

    def _rerank(
        self, query: str, ordered: tuple[ScoredChunk, ...], timings: dict[str, float]
    ) -> tuple[tuple[ScoredChunk, ...], dict[str, float], float]:
        """Rescore the top ``top_n_in`` candidates and re-order them.

        Truncation to ``top_k`` happens after this, never before -- truncating
        first would discard exactly the candidate the reranker exists to
        promote. Candidates beyond ``top_n_in`` are dropped: a cross-encoder
        costs one forward pass per candidate, and ``top_n_in`` is that budget.
        """
        if self.reranker is None:  # pragma: no cover - guarded by the caller
            return ordered, {}, 0.0

        candidates = ordered[: self._config.reranker.top_n_in]
        passages = [self._chunks[candidate.chunk_id].text for candidate in candidates]

        started = perf_counter()
        result = self.reranker.score(query, passages)
        wall_ms = (perf_counter() - started) * 1000.0

        if len(result.scores) != len(candidates):
            raise RetrieverError(
                f"reranker returned {len(result.scores)} scores for "
                f"{len(candidates)} candidates; every candidate needs exactly one"
            )

        scores = {
            candidate.chunk_id: score
            for candidate, score in zip(candidates, result.scores, strict=True)
        }
        reranked = order_candidates(
            ScoredChunk(chunk_id=chunk_id, score=score) for chunk_id, score in scores.items()
        )
        timings["rerank"] = result.compute_ms
        return reranked, scores, max(0.0, result.compute_ms - wall_ms)

    def _lexical_and_fused(
        self, query: str, dense: tuple[ScoredChunk, ...], timings: dict[str, float]
    ) -> tuple[tuple[ScoredChunk, ...], tuple[ScoredChunk, ...]]:
        """Run the lexical arm and fuse, or pass the dense ranking straight through."""
        if self.lexical_index is None or self._config.lexical is None:
            return (), dense

        with _timed(timings, "lexical"):
            lexical = self.lexical_index.search(query, self._config.lexical.top_n)

        fusion = self._config.fusion
        if fusion is None:  # pragma: no cover - RetrievalConfig forbids this pairing
            raise RetrieverError(f"config {self._config.name!r} has a lexical arm but no fusion")
        with _timed(timings, "fuse"):
            fused = reciprocal_rank_fusion([dense, lexical], k=fusion.k)

        return lexical, fused

    def _assemble(
        self,
        selected: tuple[ScoredChunk, ...],
        dense: tuple[ScoredChunk, ...],
        lexical: tuple[ScoredChunk, ...],
        fused_scores: Mapping[str, float],
        rerank_scores: Mapping[str, float],
    ) -> tuple[RetrievedPassage, ...]:
        """Turn ranked candidates into passages carrying every arm's score."""
        hybrid = self._config.retrieval_mode == "hybrid"
        dense_scores = {candidate.chunk_id: candidate.score for candidate in dense}
        lexical_scores = {candidate.chunk_id: candidate.score for candidate in lexical}

        passages = []
        for rank, candidate in enumerate(selected, start=1):
            chunk = self._chunks[candidate.chunk_id]
            passages.append(
                RetrievedPassage(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    rank=rank,
                    scores=PassageScores(
                        dense=dense_scores.get(candidate.chunk_id),
                        lexical=lexical_scores.get(candidate.chunk_id) if hybrid else None,
                        # In dense mode the ordering *is* the dense score, so
                        # a separate `fused` number would be the same value
                        # wearing a different name. Read from the map rather
                        # than `candidate.score`, which after reranking is
                        # the rerank score.
                        fused=fused_scores.get(candidate.chunk_id) if hybrid else None,
                        rerank=rerank_scores.get(candidate.chunk_id),
                    ),
                )
            )
        return tuple(passages)
