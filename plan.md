# Groundtruth — Retrieval Evaluation Harness

> A regression-tested evaluation harness for a legal research RAG assistant. Proves that a retrieval change is an improvement instead of asserting it.

**Level:** Foundational | **Effort:** 1 weekend (~10–14 focused hours) | **Status:** Not started

---

## 1. Problem Statement

A 60-attorney firm runs a research assistant over 20 years of case files and filings. Every week someone changes chunk size, swaps an embedding model, or adds a reranker — and nobody can say whether last month's version was actually better. In this domain, a missed precedent isn't a minor bug, it's a malpractice conversation. Opinion ("it feels better") is not acceptable evidence.

**Goal:** Build a harness that turns retrieval quality into a number, tracks that number over configurations, and fails a pull request automatically when quality regresses.

---

## 2. What This Project Proves

- You can separate **retrieval failure** from **generation failure** — most teams conflate the two and plateau at demo quality.
- You can build a golden evaluation set the right way (human-reviewed, not model-invented).
- You can wire quality gates into CI, not just a notebook you run once.
- You can state a tradeoff and a recommendation in writing, the artifact hiring managers actually read.

---

## 3. Scope

### In scope
- A golden dataset of query → relevant passage(s) pairs, human-reviewed.
- Retrieval metrics: Recall@k, MRR, nDCG — reported per query category.
- A generation-layer eval (faithfulness, answer relevance) kept **separate** from retrieval metrics.
- A comparison harness across at least 3 configurations (chunk size, hybrid vs. dense, reranker on/off).
- A CI regression gate that fails a PR on a defined recall drop.
- Written error analysis and a recommendation.

### Out of scope (explicitly, to avoid scope creep)
- Building a production-grade legal RAG UI.
- Fine-tuning any model.
- Multi-tenant auth, billing, or a real firm's actual case data (use a public legal corpus instead — see §5).
- Agentic orchestration of any kind.

---

## 4. Architecture Overview

```
                ┌─────────────────────┐
                │   Golden Dataset      │
                │  (queries + labels)   │
                └──────────┬───────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                                        │
┌───────▼────────┐                    ┌─────────▼─────────┐
│  Retrieval        │                    │   Generation         │
│  Configs          │                    │   Layer (optional     │
│  (chunk size,      │                    │   per config)         │
│   hybrid/dense,    │                    └─────────┬─────────┘
│   reranker on/off) │                              │
└───────┬────────┘                              │
        │                                        │
┌───────▼────────┐                    ┌─────────▼─────────┐
│  Retrieval Scorer  │                    │  Generation Scorer    │
│  Recall@k, MRR,     │                    │  Faithfulness,        │
│  nDCG               │                    │  Answer Relevance     │
└───────┬────────┘                    └─────────┬─────────┘
        │                                        │
        └──────────────────┬──────────────────┘
                           │
                ┌──────────▼───────────┐
                │  Comparison Report     │
                │  + CI Regression Gate  │
                └───────────────────────┘
```

---

## 5. Data Plan

- **Corpus:** Use a public legal corpus to stand in for firm case files — e.g. CourtListener/RECAP bulk opinions, or the CUAD (Contract Understanding Atticus Dataset) if you want contracts instead of case law. Pick ~500–2,000 documents, enough to make retrieval non-trivial but small enough to iterate on fast.
- **Golden set:** 100 query/passage pairs.
  - Write ~30 yourself as a domain-plausible attorney would ask them ("what is the standard for summary judgment in [X] circuit").
  - Generate ~100–150 candidates with an LLM from the corpus (have it read a passage and write a question it answers), then **manually review and correct every one** — this is the step people skip and the step that matters.
  - Tag each query with a category: e.g. `factual-lookup`, `multi-hop`, `negation/exclusion`, `procedural`, `ambiguous-terminology`.
  - Store as a versioned JSON/JSONL file, checked into the repo, never regenerated silently.

---

## 6. Tech Stack

| Component | Choice |
|---|---|
| Retrieval metrics | [Ragas](https://github.com/explodinggradients/ragas) or hand-rolled scorers (recommended to hand-roll recall/MRR/nDCG first — it's ~40 lines and you understand every number) |
| Embeddings | `text-embedding-3-small` or an open model (`bge-small-en-v1.5`) via sentence-transformers |
| Vector store | pgvector, Qdrant, or even in-memory FAISS for this scale |
| Reranker | `bge-reranker-base` (cross-encoder) or Cohere rerank API |
| Test runner | `pytest` |
| CI | GitHub Actions |
| Dataset versioning | Plain JSONL + git (no need for DVC at this scale) |
| Report output | Markdown table generated by script, committed or posted as a PR comment |

---

## 7. Implementation Plan (day-by-day)

### Day 1 — Data + baseline retrieval
1. Pull and chunk the corpus (fixed-size chunking, e.g. 512 tokens, 50 overlap, as baseline).
2. Build the golden set (§5). Budget the most time here — it's the foundation everything else is graded against.
3. Stand up one retrieval config: dense embedding + top-k cosine search.
4. Implement recall@k, MRR, nDCG scorers from scratch against the golden set.
5. Run baseline, save results to `results/baseline.json`.

### Day 2 — Comparisons + gate + writeup
6. Implement config #2: hybrid retrieval (BM25 + dense, reciprocal rank fusion).
7. Implement config #3: add a cross-encoder reranker on top of the best config so far.
8. Run all 3 configs against the golden set, per-category breakdown.
9. Add the generation-layer eval (faithfulness/answer relevance) on top of the winning retrieval config only — keep it clearly separate in the report.
10. Wire the CI gate: GitHub Actions workflow that runs `pytest`, fails if recall@10 drops >1 point vs. `results/baseline.json`.
11. Deliberately introduce a bad change (e.g. drop k from 10 to 2) on a branch, open a PR, confirm the gate fails. Screenshot/record this.
12. Write the error analysis: pull the 5 worst-performing queries, categorize why (chunking cut off context, embedding missed synonym, reranker demoted the right passage, etc).
13. Write the final recommendation doc (§9).

---

## 8. Acceptance Criteria / Ship Gate

This project is **not done** until every box below is checked, with evidence committed to the repo.

- [ ] Golden set of **100** query/passage pairs exists as a versioned file, with a documented labeling process (README section explaining how pairs were created and reviewed).
- [ ] Retrieval metrics (Recall@k, MRR, nDCG) implemented and reported **per query category**, not just as a single blended number.
- [ ] Generation-layer metrics (faithfulness, answer relevance) computed and reported **in a separate table/section** from retrieval metrics.
- [ ] **Three** configurations compared head-to-head: (1) chunk size variant, (2) hybrid vs. dense, (3) reranker on vs. off.
- [ ] A results table exists comparing all configs, plus a **written recommendation** stating which config you'd ship and why — including the tradeoff (e.g. latency/cost vs. quality).
- [ ] CI regression gate is wired in GitHub Actions and **demonstrably fails** on a deliberately bad change (link the PR/run in the README).
- [ ] Error analysis names the **5 worst queries** and gives a specific, non-generic reason for each failure.
- [ ] Repo has a clean README that a hiring manager can read in 3 minutes and understand: the problem, the method, the numbers, the recommendation.

### Definition of Done
The project is done when: a stranger can clone the repo, run one command to reproduce the comparison table, read the README and understand what was tested and what you'd recommend shipping — **without opening any code.** If any acceptance criterion above is unchecked, it is not done.

---

## 9. Deliverables

1. **GitHub repo** (public) with:
   - `README.md` — problem statement, method, results table, recommendation, how to reproduce.
   - `data/golden_set.jsonl` — the 100 labeled pairs.
   - `src/` — ingestion, retrieval configs, scorers, report generator.
   - `tests/` — pytest suite including the regression gate.
   - `.github/workflows/eval.yml` — CI workflow.
   - `results/` — JSON output per config, checked in for auditability.
   - `docs/error_analysis.md` — the 5-query writeup.
2. **The one-paragraph recommendation** (also pull this into your resume/portfolio talking points):
   > "Tested dense-only, hybrid, and hybrid+reranker retrieval on a 100-query legal golden set. Hybrid+reranker improved nDCG@10 by X% over dense-only but added Yms p95 latency. Recommend hybrid+reranker for accuracy-critical queries and dense-only as a fallback for latency-sensitive paths."
   (Fill in with your real numbers — this exact sentence structure is what an interviewer wants to hear.)

---

## 10. Risks / Known Traps (from the source brief)

- **The trap to avoid:** Eyeballing five queries, feeling good, and shipping. An interviewer can tell in one question. Every claim must trace back to the golden set and the metrics table.
- Don't let an LLM both generate *and* grade the golden set unreviewed — you'll just be measuring whether your retrieval agrees with another model's biases.
- Keep retrieval and generation metrics visually and structurally separate in the report — conflating them is the single most common mistake teams make.

---

## 11. Stretch Goals (only after §8 is fully checked)

- Add a 4th config: different embedding model entirely (e.g. OpenAI vs. open-source).
- Add a small Streamlit/HTML report viewer instead of a static markdown table.
- Track eval results over time as a simple line chart (config history), not just latest run.
- Add cost-per-query alongside quality metrics in the comparison table.

---

## 12. Suggested Repo Name & Tagline

**Repo:** `groundtruth-retrieval-eval`
**Tagline:** *"A CI-gated evaluation harness that proves a retrieval change was an improvement — not just asserts it."*