"""Deterministic metrics: free, fast and repeatable, so they can run live in an interview.

Retrieval : hit@k (any expected doc retrieved), recall@k, MRR
Answer    : fact recall (required facts present), violations (forbidden patterns), refusal check
A case passes only if every applicable check passes; 'reasons' says why it failed.
"""
import re

REFUSAL = re.compile(
    r"(can't|cannot|can not|unable to|not able to|i can only|only (?:help|assist)|outside (?:of )?(?:my|the) scope|"
    r"i'm sorry|i am sorry|sorry|don't have (?:access|information|that)|do not have (?:access|information|that)|"
    r"won't be able|will not|not able|i'm not able|isn't something i can|is not something i can)",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return text.lower().replace("–", "-").replace("—", "-").replace("’", "'")


def fact_recall(answer: str, must_include: list) -> tuple[float, list[str]]:
    """Each item is a string or a list of acceptable alternatives."""
    if not must_include:
        return 1.0, []
    text = _norm(answer)
    missing = []
    for item in must_include:
        options = item if isinstance(item, list) else [item]
        if not any(_norm(o) in text for o in options):
            missing.append(options[0])
    return 1 - len(missing) / len(must_include), missing


def violations(answer: str, patterns: list[str] | None) -> list[str]:
    return [p for p in (patterns or []) if re.search(p, answer)]


def retrieval_scores(retrieved: list[str], expected: list[str] | None) -> dict | None:
    if not expected:
        return None
    want = set(expected)
    found = [d for d in retrieved if d in want]
    mrr = next((1.0 / (rank + 1) for rank, d in enumerate(retrieved) if d in want), 0.0)
    return {"hit": 1.0 if found else 0.0, "recall": len(set(found)) / len(want), "mrr": mrr}


def score_case(case: dict, retrieved: list[str], answer: str | None) -> dict:
    result = {"retrieval": retrieval_scores(retrieved, case.get("expected_doc_ids")), "reasons": []}
    if result["retrieval"] is not None and result["retrieval"]["hit"] == 0:
        result["reasons"].append("retrieval_miss")
    if answer is not None:
        recall, missing = fact_recall(answer, case.get("must_include"))
        bad = violations(answer, case.get("must_not_regex"))
        refused = bool(REFUSAL.search(answer))
        result.update(fact_recall=recall, missing=missing, violations=bad, refused=refused)
        if missing:
            result["reasons"].append("missing_fact:" + "|".join(missing))
        if bad:
            result["reasons"].append("violation")
        if case.get("expected_behavior", "answer") == "refuse" and not refused:
            result["reasons"].append("did_not_refuse")
    result["passed"] = not result["reasons"]
    return result
