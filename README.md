<p align="center">
  <img src="assets/logo.png" alt="ClaimGuard logo" width="300">
</p>

# Groundtruth

> A regression-gated evaluation harness that **proves** a retrieval change was an
> improvement — instead of asserting it.

**Status: in progress (Phases 0–4 of 12 complete).** The corpus, chunking, the
committed embedding cache and retrieval are in place; scoring and the gate are
not yet implemented. **No results have been measured yet**, and this README will
not carry a results table until they are. See [Build status](#build-status).

---

## The problem

A 60-attorney firm runs a research assistant over 20 years of case files. Every
week someone changes a chunk size, swaps an embedding model, or adds a reranker
— and nobody can say whether last month's version was better.

In this domain a missed precedent is not a minor bug, it is a malpractice
conversation. *"It feels better"* is not evidence.

This project turns retrieval quality into a number, tracks that number across
configurations, and **fails automatically when quality regresses**.

## What it does

- Scores retrieval with **Recall@k, MRR@k and nDCG@k**, hand-rolled and
  cross-checked against an independent TREC-faithful implementation — reported
  **per query category**, not as one blended number.
- Scores generation (faithfulness, answer relevance) with an LLM judge, kept
  **structurally separate** from retrieval metrics and never gated.
- Compares **four configurations** head-to-head: chunk size, dense vs hybrid,
  reranker on/off.
- **Fails a regression gate** when a metric drops beyond tolerance, and blocks
  the obvious cheat of deleting the queries that fail.

## Results

*Not yet measured.* This section will contain the generated comparison table and
a one-paragraph ship recommendation once Phase 7 completes.

Publishing a table of numbers before the harness exists would be exactly the
failure this project argues against.

## Build status

| Phase | Status |
|---|---|
| 0 — Repo, harness, project scaffold | **done** |
| 1 — Config model + content hashing | **done** — 66 tests, 97% coverage |
| 2 — Corpus ingest + chunking | **done** — 150 opinions, 7.4M chars |
| 3 — Embedding cache + guards | **done** — 11,028 vectors, 8.5 MB |
| 4 — Retrieval (NumPy + BM25 + RRF) | **done** — 0.6 ms dense, 0.7 ms lexical |
| 5 — Golden set review (100 pairs) | not started |
| 6 — Scorers | not started |
| 7 — First numbers, freeze baseline | not started |
| 8 — Regression gate + demos | not started |
| 9–12 — Reranker, service, generation eval, report | not started |

## Running it

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Nothing else — no
API key, no model download, no GPU, no network.

```bash
uv sync --frozen --extra dev     # ~50 packages, no torch
uv run pytest                    # 405 tests
uv run gt --help
```

That works offline because the corpus snapshot and the embedding cache are
**committed** (11 MB total), and the model libraries live in an optional extra
the default install never pulls.

To run a single query by hand, add the `models` extra — an arbitrary query is by
definition not in the committed cache, so its vector has to be computed:

```bash
uv run gt search "standard for granting summary judgment" --config hybrid_512
```

Passage vectors still come from the cache; only the query is embedded. The gate
uses the cache-only embedder, which cannot compute anything at all.

### Regenerating the committed artifacts

Only needed when changing chunking, the embedding model, or the corpus
selection. This is the one path that needs credentials and network:

```bash
uv sync --frozen --extra dev --extra models   # adds CPU torch + sentence-transformers
cp .env.example .env                          # add COURTLISTENER_API_TOKEN

uv run gt corpus fetch --limit 150            # ~5 API requests
uv run gt cache build                         # ~99 min, single-threaded for determinism
uv run gt corpus verify && uv run gt cache verify
```

## How it is built

**Two storage paths, one retrieval code path.** The evaluation path uses exact
NumPy cosine and an in-process Okapi BM25 — no Docker, no database, no model —
so the gate is fast and bit-reproducible. The service path uses Postgres with
pgvector and `tsvector`. A parity test asserts both return identical top-10
results; everything above the storage primitives is shared code.

**Relevance labels are character spans, not chunk IDs.** Labels tied to chunk
IDs would be invalidated by any chunking change, making the chunk-size
comparison — the project's headline experiment — literally impossible to run.

**The embedding cache is committed and authoritative.** The gate's embedder
*raises* on a cache miss rather than falling back to the model. A silent
fallback would mix fresh and stale vectors and report plausible, confident,
wrong numbers with nothing to indicate a problem.

**Every result carries four hashes** — config, golden set, corpus, code. The
gate hard-fails when the golden set or corpus changes, which blocks the classic
cheat of "fixing" a regression by deleting the queries it fails. A config change
is only a warning, because changing the config *is* the workflow.

## How the golden set is built

100 query→passage pairs, and **every one is reviewed by a human**.

1. ~30 queries written by hand, as an attorney would phrase them.
2. ~120 generated from corpus passages as *candidates only*.
3. Each reviewed through `gt golden review` — accepted, edited, or rejected with
   a reason, with reviewer and timestamp recorded.
4. **Rejected candidates are kept**, so the rejection rate is visible and
   cherry-picking is not.

Letting a model both write and grade the set would measure whether retrieval
agrees with that model's biases, not whether it finds the right law.

## Honest limitations

Stated up front, because a harness that hides its own weaknesses is worthless:

- **The gate is not automatically enforced.** CI was removed by choice; nothing
  stops a commit that skips it. Evidence that it bites is reproducible via
  `make demo-regression` rather than a CI screenshot.
- **The headline numbers come from the NumPy path, not Postgres.** A parity test
  ties them together.
- **Postgres `tsvector` is not BM25** — it has no IDF term at all. It is never
  called BM25 here; the backend is named `pg_fts`.
- **The LLM judge is not deterministic.** Anthropic has no seed parameter.
  Reproducibility comes from a committed judgment cache, and the measured
  run-to-run σ is published alongside the metric.
- **Single annotator**, so there is no inter-annotator agreement statistic — only
  a published self-consistency rate.
- **150 documents, not 500.** Real Supreme Court opinions are long — measured
  over all 150, a mean of 11,269 tokens — so 500 would mean a ~30 MB embedding
  cache and git-LFS, which breaks one-command reproduction. What governs
  retrieval difficulty is the chunk count: **11,085** across both chunk
  configurations, or 11,028 distinct strings after de-duplication.

## Out of scope

A production legal RAG UI · fine-tuning · multi-tenant auth or billing · real
firm data · agentic orchestration.

See [plan.md](plan.md) for the original product brief.

## License

MIT
