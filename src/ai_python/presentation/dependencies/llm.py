from fastapi import Query

from ai_python.infrastructure.llm import AIClient, AIProvider, LLMFactory


def get_ai_client(
    provider: AIProvider = Query(
        default=AIProvider.GEMINI,
        description="LLM Provider to use",
    )
) -> AIClient:
    return LLMFactory.create(provider)
