"""Answer generation with a Groq-hosted LLM."""
import os
import time

from .config import ROOT


def build_system_prompt(cfg: dict, contexts: list[tuple[str, str]]) -> str:
    template = (ROOT / cfg["generation"]["prompt"]).read_text(encoding="utf-8")
    context = "\n\n".join(f"[{cid}] {text}" for cid, text in contexts) or "(no context)"
    return template.replace("{context}", context)


class Generator:
    def __init__(self):
        from groq import Groq

        if not os.getenv("GROQ_API_KEY"):
            raise SystemExit("GROQ_API_KEY is not set. Copy .env.example to .env and add your key, "
                             "or run with --retrieval-only.")
        # max_retries makes the SDK back off and retry on 429 rate limits (free tier)
        self.client = Groq(max_retries=8)

    def answer(self, cfg: dict, question: str, contexts: list[tuple[str, str]]) -> dict:
        g = cfg["generation"]
        system = build_system_prompt(cfg, contexts)
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=g["model"],
            temperature=g.get("temperature", 0),
            max_tokens=g.get("max_tokens", 300),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
        )
        llm_ms = (time.perf_counter() - t0) * 1000
        usage = resp.usage
        return {
            "answer": resp.choices[0].message.content or "",
            "llm_ms": round(llm_ms, 1),
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "finish_reason": resp.choices[0].finish_reason,
        }
