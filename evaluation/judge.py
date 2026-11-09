"""LLM-as-a-judge for the generate-ideas output.

The rules in rules.py only check the format. The judge scores what rules cannot:
natural Vietnamese, invented facts, relevance, specificity and persona fit.
"""

import json
from pathlib import Path
from string import Template

from pydantic import BaseModel, Field, ValidationError

from app.core.llm_output import extract_json
from app.infrastructure.llm.providers.base import AIClient
from app.infrastructure.llm.providers.bedrock import BedrockAIClient
from app.schemas.generate_idea import GenerateIdeaResponse
from evaluation.dataset import EvalCase

PROMPT = Template((Path(__file__).parent / "prompts" / "judge_ideas.txt").read_text(encoding="utf-8"))

# Claude, not Nova: a model tends to go easy on output written like its own
JUDGE_MODELS = {
    "haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet-4.6": "us.anthropic.claude-sonnet-4-6",
}
# Chosen with calibrate.py: Haiku 4.5 over-scored persona fit and failed to return JSON once
DEFAULT_JUDGE = "sonnet-4.6"

SCORED_CRITERIA = ("naturalness", "relevance", "specificity", "persona_fit")

MAX_RETRIES = 1


class JudgeError(Exception):
    pass


class CriterionScore(BaseModel):
    # Asked for before the score: a judge that must quote the problem scores more consistently
    issues: list[str]
    score: int = Field(ge=1, le=5)
    reason: str


class JudgeVerdict(BaseModel):
    invented_facts: list[str]
    safety_issues: list[str]
    naturalness: CriterionScore
    relevance: CriterionScore
    specificity: CriterionScore
    persona_fit: CriterionScore

    def average(self) -> float:
        return sum(getattr(self, name).score for name in SCORED_CRITERIA) / len(SCORED_CRITERIA)


def create_judge_client(name: str) -> AIClient:
    # temperature 0: the same output should get the same score on every run
    return BedrockAIClient(model_id=JUDGE_MODELS[name], temperature=0, top_p=None, max_tokens=2048)


def build_judge_prompt(case: EvalCase, response: GenerateIdeaResponse) -> str:
    ideas = "\n\n".join(
        f"Idea {index}\n"
        f"- title: {idea.title}\n"
        f"- approach_angle: {idea.approach_angle}\n"
        f"- hook_sentence: {idea.hook_sentence}\n"
        f"- reason: {idea.reason}"
        for index, idea in enumerate(response.ideas, start=1)
    )
    return PROMPT.substitute(
        **case.input.model_dump(),
        notes=case.notes or "(none)",
        ideas=ideas,
    )


async def judge(case: EvalCase, response: GenerateIdeaResponse, client: AIClient) -> JudgeVerdict:
    prompt = build_judge_prompt(case, response)

    for attempt in range(MAX_RETRIES + 1):
        raw = await client.generate(prompt) or ""

        try:
            return JudgeVerdict.model_validate(json.loads(extract_json(raw) or raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            if attempt == MAX_RETRIES:
                raise JudgeError(f"judge returned an invalid verdict: {str(exc).splitlines()[0]}") from exc

            # At temperature 0 the same prompt returns the same broken answer
            prompt += (
                "\n\nYour previous response was not valid JSON with the required structure. "
                "Return only the JSON object."
            )

    raise JudgeError("judge returned an invalid verdict")
