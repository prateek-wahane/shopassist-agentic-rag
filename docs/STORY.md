# The evidence story: golden v1 -> v2 -> drop -> diagnose -> fix -> recover

Run these steps yourself, in order, and commit after each one. Your git history plus `runs/` and `reports/`
then prove you did the work, and you can re-run any step live in an interview.

Commands assume PowerShell in the project folder with the virtual env active:
`.\.venv\Scripts\Activate.ps1`

---

## Step 0 - Build the index (once)

```powershell
python -m shopassist.cli ingest
python -m shopassist.cli golden validate v1
```

**Say:** "Products are one chunk each; policies are chunked by heading so every chunk is one rule with a stable id
like `returns#sale-items`. Golden cases point at those ids, which is what lets me measure retrieval exactly."

Commit: `git add -A; git commit -m "Index + golden v1 validated"`

---

## Step 1 - Baseline on golden v1

```powershell
python -m shopassist.cli eval --golden v1 --config v1_baseline --retrieval-only   # seconds, no key
python -m shopassist.cli eval --golden v1 --config v1_baseline                    # full pipeline, needs GROQ key
```

Record pass rate, hit@k, MRR, p50/p95 latency and tokens in `results/RESULTS.md`.

**Say:** "On my first golden set the baseline looked good: [x]% pass rate. That's exactly the trap - the set
reflected what I imagined users ask, not what they actually ask."

Commit: `git commit -am "Baseline v1_baseline on golden v1"`

---

## Step 2 - Release golden v2 from 'production' evidence

The 25 additions in `golden/golden_v2_additions.jsonl` model what real traffic revealed: customers paste SKU
codes, ask policy edge cases (sale items, worn shoes, bulky items), ask two-fact questions, try prompt
injection, and go off-topic. Each row has a `source` field (search logs, support tickets, red team).

```powershell
python -m shopassist.cli golden release v2 --base v1 --add golden/golden_v2_additions.jsonl --note "Coverage gaps from search logs, support tickets and red-team review"
python -m shopassist.cli golden diff v1 v2
```

**Say:** "Versions are immutable. v2 = v1 + 25 reviewed cases, logged in the changelog with the reason."

Commit: `git commit -am "Golden v2: +25 cases from logs, tickets, red team"`

---

## Step 3 - Re-evaluate the same system on v2: the drop

```powershell
python -m shopassist.cli eval --golden v2 --config v1_baseline --retrieval-only
python -m shopassist.cli eval --golden v2 --config v1_baseline
python -m shopassist.cli compare golden-v1_v1_baseline_full golden-v2_v1_baseline_full
```

Open the report in `reports/`. Look at **pass rate by intent** and **failures by reason**.

**Say:** "Nothing in the system changed, but pass rate fell from [a] to [b]. The comparison shows the old 30
cases still pass - the drop is entirely new cases, concentrated in sku_lookup (retrieval misses) and
adversarial/policy_edge (wrong or unsafe answers). So the golden set was stale, not the model worse."

---

## Step 4 - Diagnose with evidence

Read the failing rows in `runs/<run>/results.jsonl` (the `retrieved` and `reasons` fields).

- `retrieval_miss` on SKU questions: dense embeddings don't match exact codes like `RN-4472`; similar codes
  (RN-4471, RN-4472, TR-4471) look alike to a vector model.
- `missing_fact` / `violation` / `did_not_refuse` on adversarial and policy edge cases: the friendly prompt
  ("keep them happy") has no scope or no-invention rules.

**Say:** "Two separate root causes, so two separate fixes, each measured on its own."

---

## Step 5 - Fix 1: hybrid retrieval + rerank (one change)

```powershell
python -m shopassist.cli eval --golden v2 --config v2_hybrid_rerank --retrieval-only
python -m shopassist.cli eval --golden v2 --config v2_hybrid_rerank
python -m shopassist.cli compare golden-v2_v1_baseline_full golden-v2_v2_hybrid_rerank_full
```

**Say:** "BM25 matches exact tokens like '4472'; RRF fuses it with the dense ranking; a cross-encoder reranks
the top 20. Hit@3 went from [a] to [b] and MRR from [c] to [d]. Cost: retrieval p50 rose by [x] ms."

Commit: `git commit -am "Fix 1: hybrid BM25+dense with RRF and cross-encoder rerank"`

---

## Step 6 - Fix 2: grounded prompt (one change)

```powershell
python -m shopassist.cli eval --golden v2 --config v3_grounded_prompt
python -m shopassist.cli compare golden-v2_v2_hybrid_rerank_full golden-v2_v3_grounded_prompt_full
```

**Say:** "With retrieval fixed, the remaining failures were generation: the friendly prompt happily wrote a
pasta recipe and read 'stocks' as 'shoes'. A grounded prompt with scope, no-invention and refusal rules fixed
those - pass rate [a] -> [b]."

**Look for the regression too.** In the dry run during setup, the grounded prompt fixed out-of-scope cases but
*broke* `MH-001` ("If I buy the Nimbus 25 and the merino run socks, will shipping be free?"): the stricter
"use only the context" rule made the model say it had no shipping information, because the top 3 chunks were
products, not the free-shipping policy. If your compare report shows it under **Regressed**, that is your best
interview story: "my fix improved the average but regressed a multi-hop case; here's the evidence and the next
change" (see Step 9).

Commit: `git commit -am "Fix 2: grounded prompt with scope and refusal rules"`

---

## Step 7 - Guard it

```powershell
pytest -q                                                                  # retrieval gate, no key
python -m shopassist.cli gate --golden v2 --config v3_grounded_prompt --smoke --min-pass 0.85
```

Then demonstrate a regression: temporarily set `mode: dense` in `config/v3_grounded_prompt.yaml`, run
`pytest -q` and watch it fail with the list of missed cases; revert.

**Say:** "The gate runs in GitHub Actions on every PR; the retrieval gate needs no API key."

---

## Step 8 - Optional: LLM-judge metrics

```powershell
python -m shopassist.cli eval --golden v2 --config v3_grounded_prompt --judge
```

Adds G-Eval correctness and faithfulness (DeepEval, Groq judge). Deterministic checks are the primary
evidence because they are free and repeatable; the judge adds nuance. Spot-check 20 judge scores by hand and
note your agreement rate.

---

## Step 9 - The case that still fails (your "what next" answer)

After both fixes, `PE-003` ("I ran 50 km in my Nimbus 25 bought at full price. Can I still return them?") still
misses: the product name "Nimbus 25" pulls three product chunks into the top 3 and pushes out the
`returns#run-happy-trial` policy. `MH-001` has the same root cause (products crowd out the shipping policy).
Try one change at a time, measure each, keep the winner:

- `top_k: 5` (more context, more tokens - check avg_input_tokens in the compare report)
- `mmr: true` (diversity: stop three near-identical Nimbus chunks filling the slots)
- intent-aware retrieval: if the question is about returns, search policy chunks and product chunks separately
  and take the best of each (add a config option - your own code change)

Save it as `config/v4_<name>.yaml`, run it, compare against v3, and commit. Whichever you choose, you can
explain the trade-off with numbers.

---

## What to keep for the interview

- `runs/index.csv` - every run, with git commit
- `reports/compare__*.md` - the before/after reports
- `golden/CHANGELOG.md` and `golden diff v1 v2`
- git log showing one change per commit
- 5-minute live demo: `ask` 3 questions, run a retrieval-only eval (seconds), open a compare report
