from fastapi import Depends

from ai_python.llm.llm import get_ai_client
from ai_python.llm.providers.base import AIClient
from ai_python.service.content.generate_content import GenerateContent
from ai_python.service.idea.generate_idea import GenerateIdea


def get_generate_idea_service(
    ai_client: AIClient = Depends(get_ai_client),
) -> GenerateIdea:
    return GenerateIdea(ai_client=ai_client)


def get_generate_content_service(
    ai_client: AIClient = Depends(get_ai_client),
) -> GenerateContent:
    return GenerateContent(ai_client=ai_client)
