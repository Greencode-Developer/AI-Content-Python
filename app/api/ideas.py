from fastapi import APIRouter

from app.schemas.generate_idea import GenerateIdeaRequest, GenerateIdeaResponse
from app.service.idea.generate_idea import generate_idea_service

router = APIRouter(prefix ="/api/v1", tags=["Ideas"])


@router.post("/generate/ideas", response_model=GenerateIdeaResponse)
async def generate_idea(request: GenerateIdeaRequest,) -> GenerateIdeaResponse:
    return await generate_idea_service(request)
