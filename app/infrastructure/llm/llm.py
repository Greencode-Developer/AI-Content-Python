from enum import Enum
from fastapi import Query

from app.infrastructure.llm.providers.base import AIClient
from app.infrastructure.llm.providers.bedrock import BedrockAIClient
from app.infrastructure.llm.providers.gemini import GeminiAIClient


class AIProvider(str, Enum):
    BEDROCK = "bedrock"
    GEMINI = "gemini"

    
def get_ai_client(
    provider: AIProvider = Query(default=AIProvider.GEMINI),
) -> AIClient:
    match provider:
        case AIProvider.GEMINI:
            return GeminiAIClient()
        case AIProvider.BEDROCK:
            return BedrockAIClient()