# Groundtruth

> A regression-gated evaluation harness that **proves** a retrieval change was an
> improvement — instead of asserting it.

**Status: in progress (Phase 1 of 12 complete).** Documentation, project
structure and the configuration layer are in place; retrieval and scoring are
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
| 0 — Repo, ECC harness, docs, ADRs | **done** |
| 1 — Config model + content hashing | **done** — 66 tests, 97% coverage |
| 2 — Corpus ingest + chunking | not started |
| 3 — Embedding cache + guards | not started |
| 4 — Retrieval (NumPy + BM25 + RRF) | not started |
| 5 — Golden set review (100 pairs) | not started |
| 6 — Scorers | not started |
| 7 — First numbers, freeze baseline | not started |
| 8 — Regression gate + demos | not started |
| 9–12 — Reranker, service, generation eval, report | not started |

## Reproducing

Once implemented, the whole comparison reproduces with:

```bash
uv sync --frozen --extra dev
make reproduce
```

No API key, no model download, no GPU, no network. That is deliberate: the
embedding cache is committed and the model libraries are an optional extra
([ADR-0003](docs/adr/0003-committed-embedding-cache.md)).

Full instructions, including the Docker service path: **[docs/how-to-run.md](docs/how-to-run.md)**.

## How the golden set is built

100 query→passage pairs, and **every one is reviewed by a human**.

1. ~30 queries written by hand, as an attorney would phrase them.
2. ~120 generated from corpus passages as *candidates only*.
3. Each reviewed through `gt golden review` — accepted, edited, or rejected with
   a reason, with reviewer and timestamp recorded.
4. **Rejected candidates are kept**, so the rejection rate is visible and
   cherry-picking is not.

Letting a model both write and grade the set would measure whether retrieval
agrees with that model's biases. See
[ADR-0009](docs/adr/0009-golden-set-is-human-reviewed.md) and
[docs/methodology.md](docs/methodology.md).

## Honest limitations

Stated up front, because a harness that hides its own weaknesses is worthless:

- **The gate is not automatically enforced.** CI was removed by choice
  ([ADR-0006](docs/adr/0006-gate-runs-locally-not-in-ci.md)); nothing stops a
  commit that skips it. Evidence that it bites is reproducible via
  `make demo-regression` rather than a CI screenshot.
- **The headline numbers come from the NumPy path, not Postgres.** A parity test
  ties them together ([ADR-0001](docs/adr/0001-two-path-retrieval-architecture.md)).
- **Postgres `tsvector` is not BM25** — it has no IDF. It is never called BM25
  here ([ADR-0007](docs/adr/0007-pg-fts-is-not-bm25.md)).
- **The LLM judge is not deterministic.** Anthropic has no seed parameter.
  Reproducibility comes from a committed judgment cache, and the measured
  run-to-run σ is published ([ADR-0011](docs/adr/0011-claude-judges-but-never-labels.md)).
- **Single annotator**, so there is no inter-annotator agreement statistic — only
  a published self-consistency rate.

## Out of scope

A production legal RAG UI · fine-tuning · multi-tenant auth or billing · real
firm data · agentic orchestration.

## Documentation

| Document | What it covers |
|---|---|
| [docs/adr/](docs/adr/README.md) | 11 architecture decision records |
| [docs/tech-stack.md](docs/tech-stack.md) | Every dependency, and what it beat |
| [docs/how-to-run.md](docs/how-to-run.md) | Setup, gate, service, troubleshooting |
| [docs/api.md](docs/api.md) | API contract and intent |
| [docs/architecture.md](docs/architecture.md) | The two-path split and data flow |
| [docs/methodology.md](docs/methodology.md) | Labeling protocol, metric formulas, judge σ |
| [docs/harness.md](docs/harness.md) | ECC development harness: install, hooks, known gaps |
| [plan.md](plan.md) | The original product brief |

## License

MIT
