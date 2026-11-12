import asyncio
import json

import pytest

from app.core.exception.exception import InvalidLLMOutputError
from app.schemas.generate_idea import IDEA_JSON_SCHEMA, GenerateIdeaRequest
from app.service.idea import generate_idea as service

REQUEST = GenerateIdeaRequest(
    topic="Tết sale", brand_name="Shop A", tone="thân thiện", persona="", content_pillar=""
)


def ideas_json(count: int = 3, title: str = "Ý tưởng") -> str:
    ideas = [
        {"title": f"{title} {i}", "approach_angle": "góc", "hook_sentence": "hook", "reason": "lý do"}
        for i in range(count)
    ]
    return json.dumps({"ideas": ideas}, ensure_ascii=False)


class FakeAIClient:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.schemas: list[dict | None] = []

    async def generate(self, prompt: str, json_schema: dict | None = None) -> str:
        self.prompts.append(prompt)
        self.schemas.append(json_schema)
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def static_prompt(monkeypatch):
    # Keep the tests offline: the real prompt comes from Langfuse
    monkeypatch.setattr(service, "build_generate_idea_prompt", lambda request: "PROMPT")


def generate(client: FakeAIClient):
    return asyncio.run(service.generate_idea_service(REQUEST, client))


def test_parses_plain_json():
    response = generate(FakeAIClient(ideas_json()))

    assert [idea.title for idea in response.ideas] == ["Ý tưởng 0", "Ý tưởng 1", "Ý tưởng 2"]


def test_parses_json_wrapped_in_markdown_fence():
    client = FakeAIClient(f"```json\n{ideas_json()}\n```")

    response = generate(client)

    assert len(response.ideas) == 3
    assert len(client.prompts) == 1


def test_retries_with_correction_when_output_is_not_json():
    client = FakeAIClient("not json", ideas_json())

    response = generate(client)

    assert len(response.ideas) == 3
    assert client.prompts[0] == "PROMPT"
    assert "did not satisfy the required JSON format" in client.prompts[1]


@pytest.mark.parametrize("bad", [ideas_json(count=2), ideas_json(count=4), ideas_json(title="x" * 256)])
def test_retries_when_output_breaks_the_schema(bad):
    client = FakeAIClient(bad, ideas_json())

    generate(client)

    assert len(client.prompts) == 2


def test_raises_after_last_retry():
    client = FakeAIClient("bad", "bad", "bad")

    with pytest.raises(InvalidLLMOutputError):
        generate(client)

    assert len(client.prompts) == service.MAX_RETRIES + 1


def test_asks_the_provider_for_json_with_the_idea_schema():
    client = FakeAIClient(ideas_json())

    generate(client)

    assert client.schemas == [IDEA_JSON_SCHEMA]
