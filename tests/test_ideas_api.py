import pytest
from fastapi.testclient import TestClient

from app.core.exception.exception import InvalidLLMOutputError
from app.infrastructure.llm.llm import get_ai_client
from app.main import app
from app.schemas.generate_idea import GenerateIdeaResponse

# Same body the Java worker (PythonAiClient) sends, extra ids included
JAVA_REQUEST = {
    "topic": "Tết sale",
    "brand_name": "Shop A",
    "tone": "thân thiện",
    "persona": "",
    "content_pillar": "",
    "persona_id": None,
    "pillar_id": None,
    "trend_signal_id": None,
    "user_id": 1,
    "job_id": 10,
}

IDEAS = {
    "ideas": [
        {"title": f"Ý tưởng {i}", "approach_angle": "góc", "hook_sentence": "hook", "reason": "lý do"}
        for i in range(3)
    ]
}

fake_client = object()


@pytest.fixture
def client():
    app.dependency_overrides[get_ai_client] = lambda: fake_client
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_generate_ideas_uses_the_path_the_java_worker_calls(client, monkeypatch):
    received = {}

    async def fake_service(request, ai_client):
        received.update(request=request, ai_client=ai_client)
        return GenerateIdeaResponse.model_validate(IDEAS)

    monkeypatch.setattr("app.api.ideas.generate_idea_service", fake_service)

    response = client.post("/api/v1/ideas/generate?provider=bedrock", json=JAVA_REQUEST)

    assert response.status_code == 200
    assert response.json() == IDEAS
    assert received["ai_client"] is fake_client
    assert received["request"].topic == "Tết sale"


def test_invalid_llm_output_returns_502_with_detail(client, monkeypatch):
    async def failing_service(request, ai_client):
        raise InvalidLLMOutputError("LLM returned an invalid idea response")

    monkeypatch.setattr("app.api.ideas.generate_idea_service", failing_service)

    response = client.post("/api/v1/ideas/generate?provider=bedrock", json=JAVA_REQUEST)

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM returned an invalid idea response"}


def test_unknown_provider_is_rejected():
    response = TestClient(app).post("/api/v1/ideas/generate?provider=openai", json=JAVA_REQUEST)

    assert response.status_code == 422
