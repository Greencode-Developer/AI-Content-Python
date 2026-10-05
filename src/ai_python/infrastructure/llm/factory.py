from ai_python.infrastructure.llm.base import AIClient
from ai_python.infrastructure.llm.enums import AIProvider
from ai_python.infrastructure.llm.providers.bedrock import BedrockAIClient
from ai_python.infrastructure.llm.providers.gemini import GeminiAIClient


class LLMFactory:
    @staticmethod
    def create(provider: AIProvider) -> AIClient:
        match provider:
            case AIProvider.BEDROCK:
                return BedrockAIClient()
            case AIProvider.GEMINI:
                return GeminiAIClient()
            case _:
                raise ValueError(f"Unsupported AI Provider: {provider}")
