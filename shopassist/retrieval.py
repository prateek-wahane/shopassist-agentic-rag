"""Retrieval strategies: dense, sparse (BM25), hybrid (RRF), cross-encoder rerank, MMR."""
import json
import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from .index import EMBED_MODEL, INDEX_DIR

# bge models expect this instruction on queries (not on documents)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize(text: str) -> list[str]:
    # "RN-4472?" -> ["rn", "4472"]: keeps SKU parts as exact-match tokens for BM25
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        if not (INDEX_DIR / "chunks.json").exists():
            raise SystemExit("Index not found. Run: python -m shopassist.cli ingest")
        self.chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        self.ids = [c["id"] for c in self.chunks]
        self.pos = {cid: i for i, cid in enumerate(self.ids)}
        self.text = {c["id"]: c["text"] for c in self.chunks}
        self.vectors = np.load(INDEX_DIR / "vectors.npy")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANK_MODEL)
        return self._reranker

    def dense(self, query: str, k: int) -> list[str]:
        qv = self.embedder.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        scores = self.vectors @ qv  # cosine similarity (vectors are normalised)
        return [self.ids[i] for i in np.argsort(-scores)[:k]]

    def sparse(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        return [self.ids[i] for i in np.argsort(-scores)[:k] if scores[i] > 0]

    @staticmethod
    def rrf(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
        """Reciprocal Rank Fusion: score = sum of 1 / (k + rank) across lists."""
        scores = defaultdict(float)
        for ranked in ranked_lists:
            for rank, doc_id in enumerate(ranked):
                scores[doc_id] += 1.0 / (k + rank + 1)
        return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]

    def rerank(self, query: str, ids: list[str], k: int) -> list[str]:
        scores = self.reranker.predict([(query, self.text[i]) for i in ids])
        return [ids[i] for i in np.argsort(-scores)[:k]]

    def mmr(self, query: str, ids: list[str], k: int, lam: float) -> list[str]:
        """Maximal Marginal Relevance: relevant to the query but different from what's already picked."""
        vecs = self.vectors[[self.pos[i] for i in ids]]
        qv = self.embedder.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        relevance = vecs @ qv
        chosen: list[int] = []
        while len(chosen) < min(k, len(ids)):
            def score(i):
                redundancy = max((float(vecs[i] @ vecs[j]) for j in chosen), default=0.0)
                return lam * relevance[i] - (1 - lam) * redundancy
            chosen.append(max((i for i in range(len(ids)) if i not in chosen), key=score))
        return [ids[i] for i in chosen]

    def retrieve(self, query: str, cfg: dict) -> list[str]:
        r = cfg["retrieval"]
        k, pool = r["top_k"], r.get("candidates", 20)
        if r["mode"] == "dense":
            candidates = self.dense(query, pool)
        elif r["mode"] == "sparse":
            candidates = self.sparse(query, pool)
        elif r["mode"] == "hybrid":
            candidates = self.rrf([self.dense(query, pool), self.sparse(query, pool)])[:pool]
        else:
            raise SystemExit(f"Unknown retrieval mode: {r['mode']}")
        if r.get("rerank"):
            candidates = self.rerank(query, candidates, max(k, 10) if r.get("mmr") else k)
        if r.get("mmr"):
            candidates = self.mmr(query, candidates, k, r.get("mmr_lambda", 0.6))
        return candidates[:k]
