"""Run a config against a golden dataset version and save the run as evidence.

Each run writes runs/<run_id>/ with:
  config.yaml   - exact config used
  results.jsonl - one row per case: retrieved ids, answer, scores, failure reasons, latency, tokens
  summary.json  - aggregate metrics, per-intent pass rates, git commit
and appends one line to runs/index.csv (the evidence trail).
"""
import csv
import json
import shutil
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

from .config import ROOT, RUNS_DIR, load_config, load_golden
from .metrics import score_case


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                               text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "no-git"


def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def _pct(values, q):
    return round(float(np.percentile(values, q)), 1) if values else None


def summarise(rows: list[dict], cfg: dict, retrieval_only: bool) -> dict:
    scored = [r for r in rows if r["score"]["passed"] is not None]
    ret = [r["score"]["retrieval"] for r in rows if r["score"]["retrieval"]]
    by_intent = defaultdict(list)
    for r in scored:
        by_intent[r["intent"]].append(r["score"]["passed"])
    summary = {
        "n_cases": len(rows),
        "n_scored": len(scored),
        "pass_rate": _mean([1.0 if r["score"]["passed"] else 0.0 for r in scored]),
        "retrieval_hit_at_k": _mean([x["hit"] for x in ret]),
        "retrieval_recall_at_k": _mean([x["recall"] for x in ret]),
        "retrieval_mrr": _mean([x["mrr"] for x in ret]),
        "retrieval_p50_ms": _pct([r["retrieval_ms"] for r in rows], 50),
        "pass_rate_by_intent": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_intent.items())},
        "failures_by_reason": {},
    }
    reasons = defaultdict(int)
    for r in scored:
        for reason in r["score"]["reasons"]:
            reasons[reason.split(":")[0]] += 1
    summary["failures_by_reason"] = dict(sorted(reasons.items()))
    if not retrieval_only:
        g = cfg["generation"]
        tin = [r["input_tokens"] for r in rows]
        tout = [r["output_tokens"] for r in rows]
        total = [r["retrieval_ms"] + r["llm_ms"] for r in rows]
        summary.update({
            "fact_recall": _mean([r["score"].get("fact_recall") for r in rows]),
            "violations": sum(1 for r in rows if r["score"].get("violations")),
            "refusal_accuracy": _mean([1.0 if r["score"].get("refused") else 0.0 for r in rows
                                       if r.get("expected_behavior") == "refuse"]),
            "latency_p50_ms": _pct(total, 50),
            "latency_p95_ms": _pct(total, 95),
            "avg_input_tokens": round(statistics.mean(tin)),
            "avg_output_tokens": round(statistics.mean(tout)),
            "cost_per_100_queries_usd": round(100 * statistics.mean(
                [i / 1e6 * g.get("price_in_per_m", 0) + o / 1e6 * g.get("price_out_per_m", 0)
                 for i, o in zip(tin, tout)]), 4),
            "truncated_answers": sum(1 for r in rows if r.get("finish_reason") == "length"),
        })
        judged = [r["judge"] for r in rows if r.get("judge")]
        if judged:
            summary["judge_correctness"] = _mean([j.get("correctness") for j in judged])
            summary["judge_faithfulness"] = _mean([j.get("faithfulness") for j in judged])
    return summary


def run_eval(golden: str, config: str, retrieval_only: bool = False, judge: bool = False,
             limit: int | None = None, intent: str | None = None, smoke: bool = False) -> str:
    from dotenv import load_dotenv

    from .retrieval import Retriever

    load_dotenv(ROOT / ".env")
    cfg = load_config(config)
    cases = load_golden(golden)
    if intent:
        cases = [c for c in cases if c["intent"] == intent]
    if smoke:
        cases = [c for c in cases if c.get("smoke")]
    if retrieval_only:
        cases = [c for c in cases if c.get("expected_doc_ids")]
    if limit:
        cases = cases[:limit]

    retriever = Retriever()
    generator = None
    if not retrieval_only:
        from .generate import Generator
        generator = Generator()

    mode = "retrieval" if retrieval_only else "full"
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_golden-{golden}_{cfg['name']}_{mode}"
    print(f"Run {run_id}: {len(cases)} cases")
    rows = []
    for i, case in enumerate(cases, 1):
        t0 = time.perf_counter()
        retrieved = retriever.retrieve(case["input"], cfg)
        row = {"id": case["id"], "intent": case["intent"], "input": case["input"],
               "expected_behavior": case.get("expected_behavior", "answer"),
               "retrieved": retrieved, "retrieval_ms": round((time.perf_counter() - t0) * 1000, 1)}
        answer = None
        if generator:
            contexts = [(cid, retriever.text[cid]) for cid in retrieved]
            row.update(generator.answer(cfg, case["input"], contexts))
            answer = row["answer"]
            if judge:
                from .judge import judge_case
                row["judge"] = judge_case(case, answer, [t for _, t in contexts])
        row["score"] = score_case(case, retrieved, answer)
        rows.append(row)
        status = "PASS" if row["score"]["passed"] else "FAIL " + ",".join(row["score"]["reasons"])
        print(f"  [{i:>2}/{len(cases)}] {case['id']:<8} {status}")

    summary = summarise(rows, cfg, retrieval_only)
    summary.update({"run_id": run_id, "golden": golden, "config": cfg["name"], "mode": mode,
                    "judge": judge, "git": git_commit(), "timestamp": datetime.now().isoformat(timespec="seconds")})

    out = RUNS_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / cfg["_path"], out / "config.yaml")
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _append_index(summary)
    _print_summary(summary)
    return run_id


INDEX_FIELDS = ["run_id", "timestamp", "golden", "config", "mode", "git", "n_scored", "pass_rate",
                "retrieval_hit_at_k", "retrieval_mrr", "fact_recall", "violations",
                "latency_p50_ms", "latency_p95_ms", "avg_input_tokens", "cost_per_100_queries_usd"]


def _append_index(summary: dict) -> None:
    path = RUNS_DIR / "index.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(summary)


def _print_summary(s: dict) -> None:
    print("\nSummary")
    for key in ["golden", "config", "mode", "n_scored", "pass_rate", "retrieval_hit_at_k", "retrieval_mrr",
                "fact_recall", "violations", "refusal_accuracy", "latency_p50_ms", "latency_p95_ms",
                "avg_input_tokens", "avg_output_tokens", "cost_per_100_queries_usd",
                "judge_correctness", "judge_faithfulness"]:
        if s.get(key) is not None:
            print(f"  {key:<26} {s[key]}")
    print("  pass_rate_by_intent")
    for k, v in s["pass_rate_by_intent"].items():
        print(f"    {k:<24} {v}")
    if s["failures_by_reason"]:
        print("  failures_by_reason       ", s["failures_by_reason"])
    print(f"\nSaved to runs/{s['run_id']}/")
