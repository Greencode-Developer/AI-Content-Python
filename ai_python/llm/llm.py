from fastapi import Query

from ai_python.llm.enum.enums import AIProvider
from ai_python.llm.factory import LLMFactory
from ai_python.llm.providers.base import AIClient


def get_ai_client(
    provider: AIProvider = Query(
        default=AIProvider.GEMINI,
        description="LLM Provider to use",
    )
) -> AIClient:
    return LLMFactory.create(provider)
