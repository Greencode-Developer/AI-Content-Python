import asyncio
import json

from app.schemas.generate_idea import GenerateIdeaRequest
from evaluation.dataset import EvalCase
from app.infrastructure.llm.llm import AIProvider
from evaluation.run import run_experiment, save_results, summarize

REQUEST = GenerateIdeaRequest(topic="Tết", brand_name="Shop A", tone="vui", persona="", content_pillar="")

GOOD = json.dumps(
    {"ideas": [
        {"title": title, "approach_angle": "Góc tiếp cận", "hook_sentence": "Câu mở đầu", "reason": "Lý do phù hợp"}
        for title in ["Quà Tết cho mẹ", "Hậu trường đóng gói", "Mini game lì xì"]
    ]},
    ensure_ascii=False,
)


class FakeAIClient:
    def __init__(self, *responses: str | Exception | None) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def cases(count: int) -> list[EvalCase]:
    return [EvalCase(id=f"case-{i}", category="normal", input=REQUEST) for i in range(count)]


def run(client: FakeAIClient, count: int):
    # Static prompt keeps the test offline: the real prompt comes from Langfuse
    return asyncio.run(run_experiment(cases(count), client, build_prompt=lambda request: "PROMPT"))


def test_scores_each_case_with_a_single_call():
    client = FakeAIClient(GOOD, "not json")

    results = run(client, 2)

    assert client.calls == 2
    assert [result.score for result in results] == [1.0, 0.0]


def test_provider_error_is_recorded_and_the_run_continues():
    results = run(FakeAIClient(RuntimeError("Too many tokens per day"), GOOD), 2)

    assert results[0].error == "RuntimeError: Too many tokens per day"
    assert results[0].rules == []
    assert results[1].score == 1.0

    summary = summarize(results)
    assert "scored: 1, errors: 1" in summary
    assert "Too many tokens per day" in summary


def test_blocked_answer_without_text_is_an_error_not_a_crash():
    results = run(FakeAIClient(None, GOOD), 2)

    assert results[0].error == "provider returned no text"
    assert results[1].score == 1.0


def test_summary_lists_failed_rules():
    summary = summarize(run(FakeAIClient(f"```json\n{GOOD}\n```"), 1))

    assert "case-0: json_only (wrapped in a code fence)" in summary


def test_results_are_saved_as_json(tmp_path):
    path = save_results(run(FakeAIClient(GOOD), 1), AIProvider.BEDROCK, output_dir=tmp_path)

    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["provider"] == "bedrock"
    assert report["model"] == "amazon-nova-micro-v1"
    assert report["cases"][0]["rules"][0] == {"name": "valid_json", "passed": True, "detail": ""}
