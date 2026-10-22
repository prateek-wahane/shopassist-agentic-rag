"""Golden dataset governance: validate, diff and release versions."""
import json
from collections import Counter
from datetime import date

from .config import GOLDEN_DIR, ROOT, golden_path, load_golden, load_jsonl
from .index import INDEX_DIR

REQUIRED = ["id", "intent", "input", "expected_output", "expected_doc_ids", "must_include", "version_added"]


def validate(version: str) -> bool:
    cases = load_golden(version)
    errors, warnings = [], []
    known_docs = set()
    if (INDEX_DIR / "chunks.json").exists():
        known_docs = {c["id"] for c in json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))}
    else:
        warnings.append("index not built: doc ids not checked (run 'ingest' first)")

    ids = Counter(c.get("id") for c in cases)
    errors += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    inputs = Counter(c.get("input", "").strip().lower() for c in cases)
    errors += [f"duplicate question: {q!r}" for q, n in inputs.items() if n > 1]
    for c in cases:
        cid = c.get("id", "?")
        errors += [f"{cid}: missing field '{f}'" for f in REQUIRED if f not in c]
        behaviour = c.get("expected_behavior", "answer")
        if behaviour not in ("answer", "refuse"):
            errors.append(f"{cid}: expected_behavior must be 'answer' or 'refuse'")
        if behaviour == "answer" and not c.get("must_include"):
            warnings.append(f"{cid}: answer case with no must_include facts (only the judge can score it)")
        for d in c.get("expected_doc_ids", []):
            if known_docs and d not in known_docs:
                errors.append(f"{cid}: expected doc '{d}' not in index")

    print(f"Golden {version}: {len(cases)} cases")
    for intent, n in sorted(Counter(c["intent"] for c in cases).items()):
        print(f"  {intent:<16} {n}")
    print(f"  refuse cases     {sum(c.get('expected_behavior') == 'refuse' for c in cases)}")
    print(f"  smoke cases      {sum(bool(c.get('smoke')) for c in cases)}")
    for w in warnings:
        print("  WARN ", w)
    for e in errors:
        print("  ERROR", e)
    print("VALID" if not errors else f"INVALID: {len(errors)} errors")
    return not errors


def diff(old: str, new: str) -> None:
    a = {c["id"]: c for c in load_golden(old)}
    b = {c["id"]: c for c in load_golden(new)}
    added, removed = sorted(set(b) - set(a)), sorted(set(a) - set(b))
    changed = sorted(i for i in set(a) & set(b) if a[i] != b[i])
    print(f"Golden {old} -> {new}: {len(a)} -> {len(b)} cases")
    print(f"  added {len(added)}, removed {len(removed)}, changed {len(changed)}")
    by_intent = Counter(b[i]["intent"] for i in added)
    for intent, n in sorted(by_intent.items()):
        print(f"    + {n:>2} {intent}")
    for i in removed:
        print(f"    - {i}")
    for i in changed:
        fields = [k for k in set(a[i]) | set(b[i]) if a[i].get(k) != b[i].get(k)]
        print(f"    ~ {i}: {', '.join(sorted(fields))}")


def release(new: str, base: str, additions: str, note: str) -> None:
    """Create golden_<new> = golden_<base> + additions file, then validate and log it in the changelog."""
    target = golden_path(new)
    if target.exists():
        raise SystemExit(f"{target.name} already exists. Versions are immutable; release a new version instead.")
    cases = load_golden(base)
    extra = load_jsonl(ROOT / additions)
    clash = {c["id"] for c in cases} & {c["id"] for c in extra}
    if clash:
        raise SystemExit(f"IDs already in {base}: {sorted(clash)}")
    with open(target, "w", encoding="utf-8") as f:
        for c in cases + extra:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {target.relative_to(ROOT)}: {len(cases)} + {len(extra)} = {len(cases) + len(extra)} cases")
    ok = validate(new)
    log = GOLDEN_DIR / "CHANGELOG.md"
    intents = Counter(c["intent"] for c in extra)
    entry = (f"\n## {new} - {date.today().isoformat()}\n\n- Base: {base}; added {len(extra)} cases "
             f"({', '.join(f'{n} {i}' for i, n in sorted(intents.items()))})\n- Reason: {note}\n")
    with open(log, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"Changelog updated. {'Ready to evaluate.' if ok else 'Fix validation errors before using it.'}")
