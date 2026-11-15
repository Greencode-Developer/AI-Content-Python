import asyncio
import json
from contextlib import contextmanager

from app.infrastructure.llm.providers import bedrock
from app.infrastructure.llm.providers.bedrock import BedrockAIClient

SCHEMA = {"type": "object", "properties": {"ideas": {"type": "array"}}}

USAGE = {"inputTokens": 120, "outputTokens": 45, "totalTokens": 165}


class FakeBoto:
    def __init__(self, content: list[dict]) -> None:
        self.content = content
        self.requests: list[dict] = []

    def converse(self, **request):
        self.requests.append(request)
        return {"output": {"message": {"content": self.content}}, "usage": USAGE}


class FakeGeneration:
    def __init__(self, **started) -> None:
        self.started = started
        self.updates: dict = {}

    def update(self, **fields) -> None:
        self.updates.update(fields)


class FakeLangfuse:
    def __init__(self) -> None:
        self.generations: list[FakeGeneration] = []

    @contextmanager
    def start_as_current_observation(self, **started):
        generation = FakeGeneration(**started)
        self.generations.append(generation)
        yield generation


def client_with(content: list[dict]) -> tuple[BedrockAIClient, FakeBoto]:
    client = BedrockAIClient()
    client._client = FakeBoto(content)
    return client, client._client


def test_plain_prompt_returns_the_text():
    client, boto = client_with([{"text": "xin chào"}])

    assert client._invoke("PROMPT", None) == ("xin chào", USAGE)
    assert "toolConfig" not in boto.requests[0]


def test_json_schema_forces_a_tool_call_and_returns_its_input_as_json():
    client, boto = client_with([{"toolUse": {"name": "return_json", "input": {"ideas": ["Ý tưởng"]}}}])

    raw, _ = client._invoke("PROMPT", SCHEMA)

    assert json.loads(raw) == {"ideas": ["Ý tưởng"]}
    # Vietnamese is kept as is, not escaped, so the rules still see Vietnamese text
    assert "Ý tưởng" in raw
    tool_config = boto.requests[0]["toolConfig"]
    assert tool_config["tools"][0]["toolSpec"]["inputSchema"] == {"json": SCHEMA}
    assert tool_config["toolChoice"] == {"tool": {"name": "return_json"}}


def test_json_schema_without_a_tool_call_returns_empty_text_so_the_caller_retries():
    client, _ = client_with([{"text": "Xin lỗi, tôi không thể"}])

    assert client._invoke("PROMPT", SCHEMA) == ("", USAGE)


def test_generate_traces_the_call_with_model_and_token_usage(monkeypatch):
    langfuse = FakeLangfuse()
    monkeypatch.setattr(bedrock, "get_client", lambda: langfuse)
    client, _ = client_with([{"text": "xin chào"}])

    text = asyncio.run(client.generate("PROMPT"))

    assert text == "xin chào"
    [generation] = langfuse.generations
    assert generation.started["as_type"] == "generation"
    assert generation.started["model"] == "amazon.nova-micro-v1:0"
    assert generation.started["input"] == "PROMPT"
    assert generation.updates == {"output": "xin chào", "usage_details": {"input": 120, "output": 45}}
