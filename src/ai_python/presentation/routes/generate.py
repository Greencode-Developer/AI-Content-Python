from fastapi import APIRouter, Depends

from ai_python.presentation.schemas.generate_idea import GenerateIdeaRequest,GenerateIdeaResponse
from ai_python.service.idea.generate_idea import GenerateIdea

router = APIRouter()

@router.post("/ideas/generate", response_model=GenerateIdeaResponse)
async def generate_idea(
    request: GenerateIdeaRequest,
    service: GenerateIdea = Depends(get_generate_idea),
):
    return await service.execute(request)