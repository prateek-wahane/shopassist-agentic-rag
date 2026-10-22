"""Command line entry point:  python -m shopassist.cli <command> ...

  ingest                                   build the index from data/
  ask "question" --config v3_grounded_prompt
  eval --golden v1 --config v1_baseline [--retrieval-only] [--judge] [--smoke] [--intent X] [--limit N]
  compare <runA> <runB>                    run ids, unique substrings, 'latest' or 'latest~1'
  runs                                     list the evidence trail (runs/index.csv)
  golden validate v2 | golden diff v1 v2 | golden release v2 --base v1 --add golden/golden_v2_additions.jsonl --note "..."
  gate --golden v2 --config v3_grounded_prompt --min-pass 0.85 [--retrieval-only] [--smoke]
"""
import argparse
import csv
import json
import sys

from .config import ROOT, RUNS_DIR


def main(argv=None):
    p = argparse.ArgumentParser(prog="shopassist")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest")

    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--config", default="v3_grounded_prompt")

    for name in ("eval", "gate"):
        e = sub.add_parser(name)
        e.add_argument("--golden", required=True)
        e.add_argument("--config", required=True)
        e.add_argument("--retrieval-only", action="store_true", help="no LLM calls; retrieval metrics only")
        e.add_argument("--judge", action="store_true", help="add DeepEval LLM-judge metrics")
        e.add_argument("--smoke", action="store_true", help="only cases tagged smoke")
        e.add_argument("--intent")
        e.add_argument("--limit", type=int)
        if name == "gate":
            e.add_argument("--min-pass", type=float, default=0.85)

    c = sub.add_parser("compare")
    c.add_argument("run_a")
    c.add_argument("run_b")

    sub.add_parser("runs")

    g = sub.add_parser("golden")
    gsub = g.add_subparsers(dest="action", required=True)
    gv = gsub.add_parser("validate")
    gv.add_argument("version")
    gd = gsub.add_parser("diff")
    gd.add_argument("old")
    gd.add_argument("new")
    gr = gsub.add_parser("release")
    gr.add_argument("version")
    gr.add_argument("--base", required=True)
    gr.add_argument("--add", required=True)
    gr.add_argument("--note", default="")

    args = p.parse_args(argv)

    if args.cmd == "ingest":
        from .index import build_index
        print(f"Indexed {build_index()} chunks into index/")

    elif args.cmd == "ask":
        from dotenv import load_dotenv
        from .config import load_config
        from .generate import Generator
        from .retrieval import Retriever
        load_dotenv(ROOT / ".env")
        cfg = load_config(args.config)
        r = Retriever()
        ids = r.retrieve(args.question, cfg)
        print("Retrieved:", ", ".join(ids))
        out = Generator().answer(cfg, args.question, [(i, r.text[i]) for i in ids])
        print("\n" + out["answer"])
        print(f"\n[{out['llm_ms']:.0f} ms, {out['input_tokens']} in / {out['output_tokens']} out tokens]")

    elif args.cmd in ("eval", "gate"):
        from .evaluate import run_eval
        run_id = run_eval(args.golden, args.config, args.retrieval_only, args.judge,
                          args.limit, args.intent, args.smoke)
        if args.cmd == "gate":
            s = json.loads((RUNS_DIR / run_id / "summary.json").read_text(encoding="utf-8"))
            ok = s["pass_rate"] is not None and s["pass_rate"] >= args.min_pass
            print(f"\nGATE {'PASSED' if ok else 'FAILED'}: pass_rate {s['pass_rate']} vs minimum {args.min_pass}")
            sys.exit(0 if ok else 1)

    elif args.cmd == "compare":
        from .compare import compare
        compare(args.run_a, args.run_b)

    elif args.cmd == "runs":
        path = RUNS_DIR / "index.csv"
        if not path.exists():
            raise SystemExit("No runs yet.")
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        cols = ["run_id", "n_scored", "pass_rate", "retrieval_hit_at_k", "retrieval_mrr", "violations",
                "latency_p95_ms", "avg_input_tokens"]
        print("  ".join(f"{c:<{60 if c == 'run_id' else 10}}" for c in cols))
        for r in rows:
            print("  ".join(f"{r.get(c, ''):<{60 if c == 'run_id' else 10}}" for c in cols))

    elif args.cmd == "golden":
        from . import golden_tools
        if args.action == "validate":
            sys.exit(0 if golden_tools.validate(args.version) else 1)
        if args.action == "diff":
            golden_tools.diff(args.old, args.new)
        if args.action == "release":
            golden_tools.release(args.version, args.base, args.add, args.note)


if __name__ == "__main__":
    main()
