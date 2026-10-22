"""Optional LLM-as-a-judge metrics with DeepEval, using a Groq model as the judge (one API key).

Use a different, ideally stronger model than the generator to limit self-preference bias.
DeepEval's API evolves; if a signature changes, check https://deepeval.com/docs.
"""
import json

JUDGE_MODEL = "openai/gpt-oss-120b"


def _make_judge():
    from deepeval.models import DeepEvalBaseLLM
    from groq import Groq

    class GroqJudge(DeepEvalBaseLLM):
        def __init__(self, model: str):
            self.model_name = model
            self.client = Groq(max_retries=8)

        def load_model(self):
            return self.client

        def generate(self, prompt: str, schema=None):
            kwargs = {"response_format": {"type": "json_object"}} if schema else {}
            resp = self.client.chat.completions.create(
                model=self.model_name, temperature=0, reasoning_effort="low",
                messages=[{"role": "user", "content": prompt + ("\nReturn only valid JSON." if schema else "")}],
                **kwargs)
            text = resp.choices[0].message.content or ""
            return schema.model_validate(json.loads(text)) if schema else text

        async def a_generate(self, prompt: str, schema=None):
            return self.generate(prompt, schema)

        def get_model_name(self):
            return self.model_name

    return GroqJudge(JUDGE_MODEL)


def judge_case(case: dict, answer: str, contexts: list[str]) -> dict:
    from deepeval.metrics import FaithfulnessMetric, GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    judge = _make_judge()
    tc = LLMTestCase(input=case["input"], actual_output=answer,
                     expected_output=case.get("expected_output"), retrieval_context=contexts or ["(none)"])
    correctness = GEval(
        name="Correctness", model=judge, threshold=0.7, async_mode=False,
        evaluation_steps=["Compare the facts in the actual output with the expected output.",
                          "Penalise wrong prices, policies, sizes or product names heavily.",
                          "If the expected output says to decline, reward a polite refusal.",
                          "Ignore differences in wording and length."],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                           LLMTestCaseParams.EXPECTED_OUTPUT])
    scores = {}
    correctness.measure(tc)
    scores["correctness"] = correctness.score
    if contexts and case.get("expected_behavior", "answer") == "answer":
        faithfulness = FaithfulnessMetric(threshold=0.8, model=judge, async_mode=False)
        faithfulness.measure(tc)
        scores["faithfulness"] = faithfulness.score
    return scores
