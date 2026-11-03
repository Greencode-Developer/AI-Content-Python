from fastapi import APIRouter, Depends

from app.schemas.generate_content import GenerateContentRequest, GenerateContentResponse
from app.service.content.generate_content import GenerateContent, generate_content_service
from app.service.idea.generate_idea import GenerateIdea

router = APIRouter(prefix ="/api/v1", tags=["Content"])

@router.post("/generate/content", response_model=GenerateContentResponse)
async def generate_content(
    request: GenerateContentRequest,
) -> GenerateContentResponse:
    return await generate_content_service(request)
