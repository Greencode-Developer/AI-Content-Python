from fastapi import APIRouter, Depends

from app.infrastructure.llm.llm import get_ai_client
from app.infrastructure.llm.providers.base import AIClient
from app.schemas.generate_content import GenerateContentRequest, GenerateContentResponse
from app.service.content.generate_content import generate_content_service

router = APIRouter(prefix="/api/v1", tags=["Content"])


@router.post("/generate/content", response_model=GenerateContentResponse)
async def generate_content(
    request: GenerateContentRequest,
    ai_client: AIClient = Depends(get_ai_client),
) -> GenerateContentResponse:
    return await generate_content_service(request, ai_client)
