import json

from app.infrastructure.llm.providers.bedrock import BedrockAIClient

SCHEMA = {"type": "object", "properties": {"ideas": {"type": "array"}}}


class FakeBoto:
    def __init__(self, content: list[dict]) -> None:
        self.content = content
        self.requests: list[dict] = []

    def converse(self, **request):
        self.requests.append(request)
        return {"output": {"message": {"content": self.content}}}


def client_with(content: list[dict]) -> tuple[BedrockAIClient, FakeBoto]:
    client = BedrockAIClient()
    client._client = FakeBoto(content)
    return client, client._client


def test_plain_prompt_returns_the_text():
    client, boto = client_with([{"text": "xin chào"}])

    assert client._invoke("PROMPT", None) == "xin chào"
    assert "toolConfig" not in boto.requests[0]


def test_json_schema_forces_a_tool_call_and_returns_its_input_as_json():
    client, boto = client_with([{"toolUse": {"name": "return_json", "input": {"ideas": ["Ý tưởng"]}}}])

    raw = client._invoke("PROMPT", SCHEMA)

    assert json.loads(raw) == {"ideas": ["Ý tưởng"]}
    # Vietnamese is kept as is, not escaped, so the rules still see Vietnamese text
    assert "Ý tưởng" in raw
    tool_config = boto.requests[0]["toolConfig"]
    assert tool_config["tools"][0]["toolSpec"]["inputSchema"] == {"json": SCHEMA}
    assert tool_config["toolChoice"] == {"tool": {"name": "return_json"}}


def test_json_schema_without_a_tool_call_returns_empty_text_so_the_caller_retries():
    client, _ = client_with([{"text": "Xin lỗi, tôi không thể"}])

    assert client._invoke("PROMPT", SCHEMA) == ""
