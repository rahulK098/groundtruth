"""score_config: the retriever + golden set -> ScoringReport wiring.

Real retriever, real mini corpus, the deterministic HashingEmbedder stand-in
-- the same fixture the retrieval end-to-end tests use. Nothing here needs a
model or the network.
"""

from __future__ import annotations

from groundtruth.config.hashing import run_fingerprint
from groundtruth.golden.models import GoldenPair, GoldenSet, Provenance, RelevanceLabel
from groundtruth.retrieval.build import build_retriever
from groundtruth.scoring.evaluate import score_config
from groundtruth.scoring.models import K_VALUES
from tests.fixtures.mini_corpus import HashingEmbedder, mini_config, mini_documents

DOCUMENTS = mini_documents()
BY_ID = {doc.doc_id: doc for doc in DOCUMENTS}
CORPUS_HASH = "sha256:fake-corpus-hash-for-tests"
CODE_VERSION = "0.0.0-test"


def golden_pair(
    query_id: str,
    doc_id: str,
    needle: str,
    query: str,
    category: str = "factual-lookup",
    *,
    gain: int = 3,
    origin: str = "human",
) -> GoldenPair:
    text = BY_ID[doc_id].text
    start = text.index(needle)
    return GoldenPair(
        query_id=query_id,
        query=query,
        category=category,
        labels=(
            RelevanceLabel(
                doc_id=doc_id,
                char_start=start,
                char_end=start + len(needle),
                gain=gain,
                quote=needle,
            ),
        ),
        provenance=Provenance(
            origin=origin,
            generator_model="test-model" if origin == "llm" else None,
            prompt_version="test-v1" if origin == "llm" else None,
            reviewer="test",
            reviewed_at="2026-01-01T00:00:00+00:00",
            review_action="accepted",
        ),
    )


GOLDEN = GoldenSet(
    pairs=(
        golden_pair(
            "q-001",
            "mini-001",
            "summary judgment. Summary judgment is appropriate only where there is no",
            "what is the standard for granting summary judgment",
        ),
        golden_pair(
            "q-002",
            "mini-002",
            "Purposeful availment is the touchstone. A defendant who deliberately reaches",
            "minimum contacts and purposeful availment by a nonresident",
            category="procedural",
        ),
        golden_pair(
            "q-003",
            "mini-003",
            "Qualified immunity shields a government official performing a discretionary",
            "when is an officer entitled to qualified immunity",
            category="negation-exclusion",
            origin="llm",
        ),
    )
)


def retriever(mode: str = "dense"):
    config = mini_config(name=f"mini_{mode}", retrieval_mode=mode, top_k=10, dense_top_n=20)
    return build_retriever(config, DOCUMENTS, HashingEmbedder())


class TestIdentity:
    def test_carries_every_hash_the_gate_needs(self):
        report = score_config(
            retriever(), GOLDEN, corpus_manifest_hash=CORPUS_HASH, code_version=CODE_VERSION
        )
        assert report.config_hash == retriever().config.config_hash
        assert report.golden_set_hash == GOLDEN.golden_set_hash
        assert report.corpus_manifest_hash == CORPUS_HASH
        assert report.code_version == CODE_VERSION
        assert report.run_fingerprint == run_fingerprint(
            config_hash=report.config_hash,
            golden_set_hash=report.golden_set_hash,
            corpus_manifest_hash=CORPUS_HASH,
            code_version=CODE_VERSION,
        )

    def test_two_identical_runs_produce_the_same_fingerprint(self):
        first = score_config(
            retriever(), GOLDEN, corpus_manifest_hash=CORPUS_HASH, code_version=CODE_VERSION
        )
        second = score_config(
            retriever(), GOLDEN, corpus_manifest_hash=CORPUS_HASH, code_version=CODE_VERSION
        )
        assert first.run_fingerprint == second.run_fingerprint


class TestCounts:
    def test_every_pair_is_counted_and_none_are_skipped(self):
        # Every fixture pair carries a gain>=2 span by construction.
        report = score_config(
            retriever(), GOLDEN, corpus_manifest_hash=CORPUS_HASH, code_version=CODE_VERSION
        )
        assert report.n_queries == 3
        assert report.n_skipped == 0


class TestOverallMetrics:
    def test_a_near_perfect_retriever_scores_recall_1_at_k_10(self):
        # The hashing embedder retrieves the right document for every one of
        # these unambiguous, single-topic queries (mirrors the end-to-end
        # retrieval test's TOPICS).
        report = score_config(
            retriever(), GOLDEN, corpus_manifest_hash=CORPUS_HASH, code_version=CODE_VERSION
        )
        assert report.overall["recall"][10].mean == 1.0
        assert report.overall["recall"][10].n == 3

    def test_metric_table_covers_every_reported_k(self):
        report = score_config(
            retriever(), GOLDEN, corpus_manifest_hash=CORPUS_HASH, code_version=CODE_VERSION
        )
        for metric in ("recall", "mrr", "ndcg"):
            assert set(report.overall[metric]) == set(K_VALUES)


class TestBreakdowns:
    def test_by_category_has_one_entry_per_category_present(self):
        report = score_config(
            retriever(), GOLDEN, corpus_manifest_hash=CORPUS_HASH, code_version=CODE_VERSION
        )
        assert set(report.by_category) == {"factual-lookup", "procedural", "negation-exclusion"}
        assert report.by_category["factual-lookup"]["recall"][10].n == 1

    def test_by_origin_splits_human_from_llm(self):
        report = score_config(
            retriever(), GOLDEN, corpus_manifest_hash=CORPUS_HASH, code_version=CODE_VERSION
        )
        assert set(report.by_origin) == {"human", "llm"}
        assert report.by_origin["human"]["recall"][10].n == 2
        assert report.by_origin["llm"]["recall"][10].n == 1


class TestLatency:
    def test_latency_is_populated_and_non_negative(self):
        report = score_config(
            retriever(), GOLDEN, corpus_manifest_hash=CORPUS_HASH, code_version=CODE_VERSION
        )
        assert report.latency.total_ms >= 0.0
        assert report.latency.embed_ms >= 0.0

    def test_dense_config_reports_zero_lexical_latency(self):
        report = score_config(
            retriever("dense"), GOLDEN, corpus_manifest_hash=CORPUS_HASH, code_version=CODE_VERSION
        )
        assert report.latency.lexical_ms == 0.0

    def test_hybrid_config_reports_nonzero_lexical_latency(self):
        report = score_config(
            retriever("hybrid"), GOLDEN, corpus_manifest_hash=CORPUS_HASH, code_version=CODE_VERSION
        )
        assert report.latency.lexical_ms >= 0.0
