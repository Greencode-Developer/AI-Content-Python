from fastapi import APIRouter, Depends

from app.infrastructure.llm.llm import get_ai_client
from app.infrastructure.llm.providers.base import AIClient
from app.schemas.generate_idea import GenerateIdeaRequest, GenerateIdeaResponse
from app.service.idea.generate_idea import generate_idea_service

router = APIRouter(prefix="/api/v1/ideas", tags=["Ideas"])


@router.post("/generate", response_model=GenerateIdeaResponse)
async def generate_idea(
    request: GenerateIdeaRequest,
    ai_client: AIClient = Depends(get_ai_client),
) -> GenerateIdeaResponse:
    return await generate_idea_service(request, ai_client)
