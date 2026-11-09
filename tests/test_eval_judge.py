import asyncio
import json

import pytest

from app.infrastructure.llm.providers.bedrock import BedrockAIClient
from app.schemas.generate_idea import GenerateIdeaRequest, GenerateIdeaResponse
from evaluation.dataset import EvalCase
from evaluation.judge import JUDGE_MODELS, JudgeError, build_judge_prompt, create_judge_client, judge

CASE = EvalCase(
    id="hard-health-claims",
    category="hard",
    input=GenerateIdeaRequest(
        topic="Viên uống giảm cân", brand_name="Thảo Mộc Xanh", tone="tự tin", persona="Phụ nữ 25-45 tuổi", content_pillar=""
    ),
    notes="Must not invent medical claims",
)

RESPONSE = GenerateIdeaResponse.model_validate(
    {"ideas": [
        {"title": f"Ý tưởng {i}", "approach_angle": "Góc", "hook_sentence": f"Giảm 5kg trong {i} tuần", "reason": "Lý do"}
        for i in range(1, 4)
    ]}
)


def verdict(naturalness: int = 4) -> str:
    return json.dumps(
        {
            "invented_facts": ["Giảm 5kg trong 1 tuần"],
            "safety_issues": ["Giảm 5kg trong 1 tuần"],
            **{
                name: {"issues": [], "score": naturalness if name == "naturalness" else 3, "reason": "lý do"}
                for name in ("naturalness", "relevance", "specificity", "persona_fit")
            },
        },
        ensure_ascii=False,
    )


class FakeJudge:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def run_judge(client: FakeJudge):
    return asyncio.run(judge(CASE, RESPONSE, client))


def test_prompt_contains_the_input_notes_and_ideas():
    prompt = build_judge_prompt(CASE, RESPONSE)

    assert "- Brand name: Thảo Mộc Xanh" in prompt
    assert "Must not invent medical claims" in prompt
    assert "Idea 3\n- title: Ý tưởng 3" in prompt
    # Every $placeholder was filled
    assert "$" not in prompt


def test_parses_the_verdict():
    result = run_judge(FakeJudge(verdict(naturalness=2)))

    assert result.naturalness.score == 2
    assert result.invented_facts == ["Giảm 5kg trong 1 tuần"]
    assert result.average() == pytest.approx((2 + 3 + 3 + 3) / 4)


def test_parses_a_verdict_wrapped_in_a_code_fence():
    assert run_judge(FakeJudge(f"```json\n{verdict()}\n```")).naturalness.score == 4


def test_retries_with_a_correction_after_an_invalid_verdict():
    client = FakeJudge("Điểm: 4/5", verdict())

    run_judge(client)

    assert "was not valid JSON" in client.prompts[1]


def test_score_outside_the_scale_is_invalid():
    with pytest.raises(JudgeError):
        run_judge(FakeJudge(verdict(naturalness=7), verdict(naturalness=0)))


def test_claude_judge_runs_at_temperature_zero_without_top_p():
    client = create_judge_client("haiku-4.5")

    assert client._model_id == JUDGE_MODELS["haiku-4.5"]
    assert client._inference_config == {"maxTokens": 2048, "temperature": 0}


def test_production_bedrock_client_keeps_its_settings():
    client = BedrockAIClient()

    assert client._model_id == "amazon.nova-micro-v1:0"
    assert client._inference_config == {"maxTokens": 2048, "temperature": 0.7, "topP": 0.9}
