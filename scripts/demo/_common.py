"""Shared plumbing for the three gate-bites demos.

Per ADR-0006: "evidence that the gate actually bites is produced
reproducibly rather than by screenshot." Each demo script in this directory
applies one deliberate change, scores it, gates it against the real frozen
baseline in results/baseline.json, and writes the transcript to
docs/evidence/.

None of the three ever writes to a tracked file. "Self-reverting" is
achieved the safe way: by never touching configs/ or data/cache/ in the
first place, rather than editing and reverting them. Every demo builds its
modified configuration and (where new chunk text needs embedding) its
embeddings entirely in memory, against the real corpus, the real golden
set, and the real committed baseline -- so the numbers are real, and the
repository is never left dirty even if a demo crashes partway through.
"""

from __future__ import annotations

import sys
from pathlib import Path

from groundtruth import __version__
from groundtruth.config.models import RetrievalConfig
from groundtruth.corpus.models import Document
from groundtruth.corpus.snapshot import read_manifest, verify_snapshot
from groundtruth.embedding.cache import CachedEmbedder
from groundtruth.embedding.store import read_store, store_dir
from groundtruth.gate.compare import ConfigGateResult, evaluate_config
from groundtruth.gate.policy import load_gate_policy
from groundtruth.golden.models import GoldenSet
from groundtruth.golden.store import read_golden_set
from groundtruth.paths import cache_dir, configs_dir, corpus_dir, golden_dir
from groundtruth.retrieval.build import build_retriever
from groundtruth.scoring.evaluate import score_config
from groundtruth.scoring.models import ScoringReport
from groundtruth.scoring.report_io import read_baseline

EVIDENCE_DIR = Path(__file__).resolve().parents[2] / "docs" / "evidence"

#: How many documents a subset-scoped demo (chunk size, overlap) searches.
#: A change that fragments or widens chunks means every string is novel --
#: no committed vector to reuse -- so scoring the full 150-document corpus
#: at, say, 64-token windows means tens of thousands of fresh CPU forward
#: passes: hours, not a demo. Bounding to a handful of real documents keeps
#: the model, the code and the metrics real while keeping the runtime real
#: too -- a few minutes, not a few hours.
SUBSET_SIZE = 6


def score_modified_config(
    config: RetrievalConfig, *, fresh_embedder: object | None = None
) -> ScoringReport:
    """Score ``config`` against the real corpus and golden set.

    ``fresh_embedder`` is supplied only by demos that change chunking (new
    chunk text means a cache miss). It wraps a real model over the
    committed cache: every string the committed cache already has is
    reused at zero cost; anything new is computed in memory and never
    written back to disk.
    """
    documents = verify_snapshot(corpus_dir())
    manifest_hash = read_manifest(corpus_dir()).manifest_hash
    golden = read_golden_set(golden_dir())

    if fresh_embedder is not None:
        embedder = fresh_embedder
    else:
        store = read_store(
            store_dir(
                cache_dir() / "embeddings", config.embedding.model_id, config.embedding.revision
            )
        )
        from groundtruth.embedding.cache import CachedOnlyEmbedder

        embedder = CachedOnlyEmbedder(store)

    retriever = build_retriever(config, documents, embedder)  # type: ignore[arg-type]
    return score_config(
        retriever, golden, corpus_manifest_hash=manifest_hash, code_version=__version__
    )


def in_memory_embedder(config: RetrievalConfig) -> CachedEmbedder:
    """A real model, backed by the committed cache, that never persists.

    Reuses every already-cached vector (the 512- and 256-token chunks, every
    golden query) at zero cost; computes anything new -- like a chunker
    change's novel windows -- with the real model, in memory only.
    """
    from groundtruth.embedding.sentence_transformer import SentenceTransformerEmbedder

    try:
        store = read_store(
            store_dir(
                cache_dir() / "embeddings", config.embedding.model_id, config.embedding.revision
            )
        )
    except Exception:
        store = None

    delegate = SentenceTransformerEmbedder(
        config.embedding.model_id, config.embedding.revision, normalize=config.embedding.normalize
    )
    return CachedEmbedder(store, delegate)


def gate_against_baseline(report: ScoringReport, config_name: str) -> ConfigGateResult:
    """Compare one real, freshly-scored report against the frozen baseline."""
    baseline = read_baseline(Path("results"))
    policy = load_gate_policy(configs_dir() / "gate_policy.yaml")
    return evaluate_config(config_name, baseline.runs, {config_name: report}, policy)


def _document_subset(documents: tuple[Document, ...], golden: GoldenSet) -> tuple[Document, ...]:
    """The first ``SUBSET_SIZE`` documents (by doc_id) that a golden label names."""
    referenced = sorted({label.doc_id for pair in golden.pairs for label in pair.labels})
    chosen = set(referenced[:SUBSET_SIZE])
    return tuple(doc for doc in documents if doc.doc_id in chosen)


def _golden_subset(golden: GoldenSet, subset_doc_ids: set[str]) -> GoldenSet:
    """Pairs whose every label falls inside the document subset.

    A pair with even one label outside the subset cannot be fairly scored --
    scoring it here would report a spurious recall of 0 for a document this
    demo never even indexed, not a real regression.
    """
    kept = tuple(
        pair
        for pair in golden.pairs
        if all(label.doc_id in subset_doc_ids for label in pair.labels)
    )
    return GoldenSet(pairs=kept)


def score_before_and_after(
    base_config: RetrievalConfig, modified_config: RetrievalConfig
) -> tuple[ScoringReport, ScoringReport, int]:
    """Score the same config before and after a chunking change, on a shared subset.

    Both sides search the identical, documented handful of real documents and
    are scored against the identical filtered golden subset -- so their
    golden_set_hash and corpus_manifest_hash always agree with each other,
    and the only thing evaluate_config can find different is the metrics.
    This is a **before/after comparison**, not a claim about
    ``results/baseline.json``: the full corpus was never searched, and the
    identity hashes recorded here reflect the subset, not the committed
    baseline's.
    """
    documents = verify_snapshot(corpus_dir())
    golden = read_golden_set(golden_dir())

    subset_documents = _document_subset(documents, golden)
    subset_doc_ids = {doc.doc_id for doc in subset_documents}
    subset_golden = _golden_subset(golden, subset_doc_ids)

    # A fabricated identity, deliberately: it is scoped to this demo's
    # subset, not to the committed corpus snapshot, so it must not be
    # mistaken for corpus_manifest_hash of the real corpus.
    subset_hash = f"demo-subset:{','.join(sorted(subset_doc_ids))}"

    before_embedder = in_memory_embedder(base_config)
    before = score_config(
        build_retriever(base_config, subset_documents, before_embedder),
        subset_golden,
        corpus_manifest_hash=subset_hash,
        code_version=__version__,
    )

    after_embedder = in_memory_embedder(modified_config)
    after = score_config(
        build_retriever(modified_config, subset_documents, after_embedder),
        subset_golden,
        corpus_manifest_hash=subset_hash,
        code_version=__version__,
    )

    return before, after, len(subset_golden.pairs)


def gate_before_vs_after(
    before: ScoringReport, after: ScoringReport, config_name: str
) -> ConfigGateResult:
    """Compare the after-report against the before-report directly.

    Uses the same evaluate_config the real gate uses, keyed under a shared
    label rather than a real config name -- this is a standalone comparison
    for demo purposes, not a claim against the committed baseline.
    """
    policy = load_gate_policy(configs_dir() / "gate_policy.yaml")
    return evaluate_config(config_name, {config_name: before}, {config_name: after}, policy)


def summarize(report: ScoringReport) -> str:
    lines = [f"config_hash : {report.config_hash}"]
    for metric in ("recall", "mrr", "ndcg"):
        summary = report.overall.get(metric, {}).get(10)
        if summary is not None:
            lines.append(f"{metric}@10     : {summary.mean:.3f}  (n={summary.n})")
    return "\n".join(lines)


def write_evidence(name: str, transcript: str) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / f"{name}.txt"
    path.write_text(transcript, encoding="utf-8")
    return path


def report_result(
    demo_name: str, baseline_summary: str, result: ConfigGateResult, expect_pass: bool
) -> int:
    """Print, capture, and return the process exit code this demo asserts."""
    lines = [
        f"=== {demo_name} ===",
        "",
        "--- current run ---",
        baseline_summary,
        "",
        f"gate status: {result.status}",
    ]
    if result.identity_failures:
        lines.append("identity failures:")
        lines.extend(f"  {f}" for f in result.identity_failures)
    if result.regressions:
        lines.append("metric regressions:")
        lines.extend(f"  {r}" for r in result.regressions)
    if not result.identity_failures and not result.regressions:
        lines.append("no regressions or identity failures")

    transcript = "\n".join(lines) + "\n"
    print(transcript)
    path = write_evidence(demo_name, transcript)
    print(f"evidence written to {path}")

    passed = result.passed
    if passed == expect_pass:
        print(f"OK: gate {'passed' if passed else 'failed'} as expected")
        return 0
    print(
        f"UNEXPECTED: gate {'passed' if passed else 'failed'}, expected "
        f"{'a pass' if expect_pass else 'a failure'}",
        file=sys.stderr,
    )
    return 1
