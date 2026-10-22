"""Compare two runs and write a Markdown evidence report: metric deltas, per-intent changes, per-case flips."""
import json
from datetime import datetime
from pathlib import Path

from .config import REPORTS_DIR, RUNS_DIR, load_jsonl

METRICS = [  # (key, label, higher_is_better)
    ("pass_rate", "Pass rate", True),
    ("retrieval_hit_at_k", "Retrieval hit@k", True),
    ("retrieval_recall_at_k", "Retrieval recall@k", True),
    ("retrieval_mrr", "Retrieval MRR", True),
    ("fact_recall", "Fact recall", True),
    ("violations", "Violations (count)", False),
    ("refusal_accuracy", "Refusal accuracy", True),
    ("judge_correctness", "Judge correctness (G-Eval)", True),
    ("judge_faithfulness", "Judge faithfulness", True),
    ("latency_p50_ms", "Latency p50 (ms)", False),
    ("latency_p95_ms", "Latency p95 (ms)", False),
    ("avg_input_tokens", "Avg input tokens", False),
    ("cost_per_100_queries_usd", "Cost per 100 queries ($)", False),
]


def resolve_run(ref: str) -> Path:
    runs = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()) if RUNS_DIR.exists() else []
    if not runs:
        raise SystemExit("No runs yet. Run an eval first.")
    if ref == "latest":
        return runs[-1]
    if ref.startswith("latest~"):
        return runs[-1 - int(ref.split("~")[1])]
    exact = [p for p in runs if p.name == ref]
    matches = exact or [p for p in runs if ref in p.name]
    if len(matches) != 1:
        names = "\n  ".join(p.name for p in matches[-10:])
        raise SystemExit(f"'{ref}' matched {len(matches)} runs. Be more specific:\n  {names}")
    return matches[0]


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _fmt(v):
    return "-" if v is None else v


def _delta(a, b, higher_better):
    if a is None or b is None:
        return "-"
    d = round(b - a, 3)
    if d == 0:
        return "0"
    better = (d > 0) == higher_better
    return f"{d:+g} {'(better)' if better else '(worse)'}"


def compare(ref_a: str, ref_b: str) -> Path:
    a_dir, b_dir = resolve_run(ref_a), resolve_run(ref_b)
    a = json.loads((a_dir / "summary.json").read_text(encoding="utf-8"))
    b = json.loads((b_dir / "summary.json").read_text(encoding="utf-8"))
    ra = {r["id"]: r for r in load_jsonl(a_dir / "results.jsonl")}
    rb = {r["id"]: r for r in load_jsonl(b_dir / "results.jsonl")}

    out = [f"# Run comparison\n",
           f"- **A (before):** `{a['run_id']}` - golden {a['golden']}, config {a['config']}, git {a['git']}",
           f"- **B (after):** `{b['run_id']}` - golden {b['golden']}, config {b['config']}, git {b['git']}",
           f"- Generated {datetime.now():%Y-%m-%d %H:%M}\n"]
    if a["golden"] != b["golden"]:
        out.append(f"> Golden versions differ ({a['golden']} vs {b['golden']}). Whole-set metrics are not "
                   f"directly comparable; see the shared-case section below.\n")

    out.append("## Metrics\n")
    rows = [(label, _fmt(a.get(k)), _fmt(b.get(k)), _delta(a.get(k), b.get(k), hib))
            for k, label, hib in METRICS if a.get(k) is not None or b.get(k) is not None]
    rows.insert(0, ("Cases scored", a["n_scored"], b["n_scored"], b["n_scored"] - a["n_scored"]))
    out.append(_table(["Metric", "A", "B", "Change"], rows))

    shared = sorted(set(ra) & set(rb))
    if shared and (len(shared) != len(ra) or len(shared) != len(rb)):
        pa = sum(ra[i]["score"]["passed"] for i in shared) / len(shared)
        pb = sum(rb[i]["score"]["passed"] for i in shared) / len(shared)
        out.append(f"\n**Shared cases only (n={len(shared)}):** pass rate {pa:.3f} -> {pb:.3f}")

    out.append("\n## Pass rate by intent\n")
    intents = sorted(set(a["pass_rate_by_intent"]) | set(b["pass_rate_by_intent"]))
    out.append(_table(["Intent", "A", "B", "Change"],
                      [(i, _fmt(a["pass_rate_by_intent"].get(i)), _fmt(b["pass_rate_by_intent"].get(i)),
                        _delta(a["pass_rate_by_intent"].get(i), b["pass_rate_by_intent"].get(i), True))
                       for i in intents]))

    fixed = [i for i in shared if not ra[i]["score"]["passed"] and rb[i]["score"]["passed"]]
    regressed = [i for i in shared if ra[i]["score"]["passed"] and not rb[i]["score"]["passed"]]
    still = [i for i in shared if not ra[i]["score"]["passed"] and not rb[i]["score"]["passed"]]
    new_ids = sorted(set(rb) - set(ra))
    new_fail = [i for i in new_ids if not rb[i]["score"]["passed"]]

    out.append("\n## Case-level changes\n")
    out.append(_table(["Bucket", "Count"], [("Fixed (fail -> pass)", len(fixed)),
                                             ("Regressed (pass -> fail)", len(regressed)),
                                             ("Still failing", len(still)),
                                             ("New cases in B", len(new_ids)),
                                             ("New cases failing in B", len(new_fail))]))

    def detail(title, ids, src_a, src_b):
        if not ids:
            return
        out.append(f"\n### {title}\n")
        rows = []
        for i in ids:
            r = src_b.get(i) or src_a.get(i)
            before = ", ".join(src_a[i]["score"]["reasons"]) if i in src_a else "(not in A)"
            after = ", ".join(src_b[i]["score"]["reasons"]) or "pass" if i in src_b else "-"
            rows.append((i, r["intent"], r["input"][:70].replace("|", "/"), before or "pass", after,
                         ", ".join(src_b[i]["retrieved"]) if i in src_b else "-"))
        out.append(_table(["Case", "Intent", "Question", "A", "B", "B retrieved"], rows))

    detail("Regressed", regressed, ra, rb)
    detail("Fixed", fixed, ra, rb)
    detail("New cases failing in B", new_fail, ra, rb)
    detail("Still failing", still, ra, rb)

    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"compare__{a['config']}-{a['golden']}__vs__{b['config']}-{b['golden']}__{b_dir.name[:15]}.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\nReport saved: {path.relative_to(path.parent.parent)}")
    return path
