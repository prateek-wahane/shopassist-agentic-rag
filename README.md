# ShopAssist - evaluation-driven RAG

A small retrieval-augmented shopping assistant for a fictional running store (StrideStore), built to practise
**measuring and improving** an LLM system with evidence: versioned golden datasets, saved runs, before/after
comparison reports and a CI gate.

## What it shows

- **Golden dataset governance**: v1 (30 cases) -> v2 (+25 cases from search logs, support tickets, red team),
  immutable versions, validation, diff and changelog (`golden/`).
- **Evaluation**: deterministic metrics that are free and repeatable (retrieval hit@k, recall@k, MRR; fact recall,
  forbidden-pattern violations, refusal accuracy; latency p50/p95, tokens, cost) plus optional DeepEval
  LLM-judge metrics (G-Eval correctness, faithfulness).
- **Improvement loop**: the same system scores lower on golden v2; per-intent and per-case reports show why;
  two targeted fixes (hybrid BM25 + dense retrieval with RRF and cross-encoder rerank, then a grounded prompt) each
  measured on their own.
- **Evidence trail**: every run saved in `runs/` with its config and git commit; comparisons in `reports/`.

## Architecture

```text
question -> retriever (dense bge-small | BM25 | hybrid RRF -> cross-encoder rerank -> optional MMR)
         -> top-k chunks -> Groq LLM with prompt template -> answer
         -> metrics vs golden case -> runs/<run_id>/ -> compare -> reports/
```

## Setup (Windows PowerShell)

```powershell
cd D:\Projects\shopassist-agentic-rag
.\.venv\Scripts\Activate.ps1                     # created during setup; otherwise: python -m venv .venv
# pip install torch --index-url https://download.pytorch.org/whl/cpu; pip install -r requirements.txt
copy .env.example .env                           # then paste your free key from console.groq.com
python -m shopassist.cli ingest
```

## Commands

| Command | What it does |
| --- | --- |
| `python -m shopassist.cli ask "Is TR-4471 waterproof?" --config v3_grounded_prompt` | One question, shows retrieved chunks, answer, latency, tokens |
| `python -m shopassist.cli eval --golden v2 --config v1_baseline --retrieval-only` | Retrieval metrics only - seconds, no API key |
| `python -m shopassist.cli eval --golden v2 --config v3_grounded_prompt` | Full pipeline with answer metrics |
| `python -m shopassist.cli eval ... --judge` | Adds DeepEval G-Eval correctness + faithfulness |
| `python -m shopassist.cli compare latest~1 latest` | Before/after report: metric deltas, intents, fixed/regressed cases |
| `python -m shopassist.cli runs` | The evidence trail |
| `python -m shopassist.cli golden validate v2` / `golden diff v1 v2` | Dataset checks and version diff |
| `python -m shopassist.cli gate --golden v2 --config v3_grounded_prompt --smoke --min-pass 0.85` | Exit code 1 if below threshold |
| `pytest -q` | Retrieval gate (no key) - used in GitHub Actions |

## The walkthrough

Follow [docs/STORY.md](docs/STORY.md): baseline on v1 -> release v2 -> drop -> diagnose -> fix 1 -> fix 2 -> gate.

## Results

Fill this table from `runs/index.csv` and the reports after you run the story (n = cases scored).

| Run | Golden | n | Pass rate | Hit@3 | MRR | Violations | p95 latency | Avg input tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1_baseline | v1 | | | | | | | |
| v1_baseline | v2 | | | | | | | |
| v2_hybrid_rerank | v2 | | | | | | | |
| v3_grounded_prompt | v2 | | | | | | | |

## Known limitations

- Small synthetic catalog (20 products, 6 policies) and 55 golden cases: results are directional; at n = 55 a
  pass-rate difference of a few points can be noise.
- Deterministic answer checks use required facts and forbidden patterns; they can miss paraphrases, which is
  why the optional LLM judge exists.
- `PE-003` still fails after both fixes - see Step 9 in the story.
