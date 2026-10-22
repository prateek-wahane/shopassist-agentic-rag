"""Fast CI gate: no API key needed. Fails the build if retrieval quality on the latest golden set drops.

Run: pytest -q   (expects 'python -m shopassist.cli ingest' to have run)
"""
import os

import pytest

from shopassist.config import load_config, load_golden
from shopassist.metrics import retrieval_scores

GOLDEN = os.getenv("GOLDEN", "v2")
CONFIG = os.getenv("CONFIG", "v3_grounded_prompt")
MIN_HIT = float(os.getenv("MIN_HIT", "0.9"))


@pytest.fixture(scope="module")
def retriever():
    from shopassist.retrieval import Retriever
    return Retriever()


def test_retrieval_hit_rate(retriever):
    cfg = load_config(CONFIG)
    cases = [c for c in load_golden(GOLDEN) if c.get("expected_doc_ids")]
    scores = [retrieval_scores(retriever.retrieve(c["input"], cfg), c["expected_doc_ids"]) for c in cases]
    misses = [c["id"] for c, s in zip(cases, scores) if s["hit"] == 0]
    hit_rate = 1 - len(misses) / len(cases)
    assert hit_rate >= MIN_HIT, f"hit@k {hit_rate:.3f} < {MIN_HIT} on golden {GOLDEN}; misses: {misses}"
