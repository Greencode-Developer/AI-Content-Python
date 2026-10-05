from fastapi import Depends

from ai_python.infrastructure.llm import AIClient
from ai_python.presentation.dependencies.llm import get_ai_client
from ai_python.service.idea.generate_idea import GenerateIdea


def get_generate_idea_service(
    ai_client: AIClient = Depends(get_ai_client),
) -> GenerateIdea:
    return GenerateIdea(ai_client=ai_client)
