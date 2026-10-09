from fastapi import APIRouter, Depends


from ai_python.schemas.generate_content import GenerateContentRequest, GenerateContentResponse
from ai_python.schemas.generate_idea import GenerateIdeaRequest, GenerateIdeaResponse
from ai_python.service.content.generate_content import GenerateContent
from ai_python.service.dependencies.services import get_generate_content_service, get_generate_idea_service
from ai_python.service.idea.generate_idea import GenerateIdea

router = APIRouter()


# ── Ideas ──────────────────────────────────────────────────────────────────

@router.post(
    "/ideas/generate",
    response_model=GenerateIdeaResponse,
    tags=["Ideas"],
    summary="Sinh ý tưởng nội dung",
    description="Dùng AI đề xuất danh sách ý tưởng dựa trên chủ đề, thương hiệu và chân dung khách hàng.",
)
async def generate_idea(
    request: GenerateIdeaRequest,
    service: GenerateIdea = Depends(get_generate_idea_service),
) -> GenerateIdeaResponse:
    return await service.execute(request)


# ── Content ────────────────────────────────────────────────────────────────

@router.post(
    "/content/generate",
    response_model=GenerateContentResponse,
    tags=["Content"],
    summary="Sinh bài đăng Facebook",
    description=(
        "Nhận thông tin thương hiệu, chân dung khách hàng, ý tưởng nguồn "
        "và định dạng bài (text / image_text / video_script), "
        "trả về caption hoàn chỉnh kèm hook, hashtag và CTA."
    ),
    status_code=200,
)
async def generate_content(
    request: GenerateContentRequest,
    service: GenerateContent = Depends(get_generate_content_service),
) -> GenerateContentResponse:
    return await service.execute(request)
