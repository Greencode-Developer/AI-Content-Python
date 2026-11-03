import json
import re

from app.infrastructure.llm.providers.base import AIClient
from app.schemas.generate_content import GenerateContentRequest, GenerateContentResponse, GeneratedContent

_UNKNOWN_MODEL = "unknown"


async def generate_content_service(
    request: GenerateContentRequest,
    ai_client: AIClient,
) -> GenerateContentResponse:
    prompt = build_prompt(request)
    raw = await ai_client.generate(prompt)

    content = parse_response(raw, request.post_format)

    model_name = getattr(type(ai_client), "MODEL_NAME", _UNKNOWN_MODEL)

    return GenerateContentResponse(
        content=content,
        ai_model_used=model_name,
        idea_id=request.idea_id,
        pillar_id=request.pillar_id,
        persona_id=request.persona_id,
        channel_id=request.channel_id,
    )


def build_prompt(req: GenerateContentRequest) -> str:
    sections: list[str] = []

    sections.append(
        "Bạn là chuyên gia viết nội dung mạng xã hội cho các shop nhỏ tại Việt Nam. "
        "Nhiệm vụ: sinh một bài đăng Facebook hoàn chỉnh theo thông tin dưới đây."
    )

    brand = req.brand
    brand_block = [
        "## THÔNG TIN THƯƠNG HIỆU",
        f"- Mô tả: {brand.description}",
        f"- Giọng điệu: {brand.tone_of_voice}",
    ]

    if brand.forbidden_words:
        brand_block.append(
            f"- Từ / cụm từ TUYỆT ĐỐI KHÔNG dùng: {brand.forbidden_words}"
        )

    sections.append("\n".join(brand_block))

    if req.persona:
        persona = req.persona
        persona_lines = [
            "## CHÂN DUNG KHÁCH HÀNG MỤC TIÊU",
            f"- Tên chân dung: {persona.name}",
        ]

        if persona.age_range:
            persona_lines.append(f"- Độ tuổi: {persona.age_range}")
        if persona.occupation:
            persona_lines.append(f"- Nghề nghiệp: {persona.occupation}")
        if persona.pain_points:
            persona_lines.append(f"- Nỗi đau / vấn đề: {persona.pain_points}")
        if persona.desires:
            persona_lines.append(f"- Mong muốn: {persona.desires}")
        if persona.typical_phrases:
            persona_lines.append(f"- Câu họ hay nói: {persona.typical_phrases}")

        sections.append("\n".join(persona_lines))

    if req.pillar_name:
        sections.append(
            f"## TRỤ CỘT NỘI DUNG\n- Trụ cột: {req.pillar_name}\n"
            "  (Bài viết phải phục vụ đúng mục tiêu của trụ cột này)"
        )

    if req.idea:
        idea_lines = [
            "## Ý TƯỞNG NGUỒN",
            f"- Tiêu đề: {req.idea.title}",
        ]

        if req.idea.approach_angle:
            idea_lines.append(f"- Góc tiếp cận: {req.idea.approach_angle}")
        if req.idea.hook_sentence:
            idea_lines.append(f"- Câu hook gợi ý: {req.idea.hook_sentence}")

        sections.append("\n".join(idea_lines))

    sections.append(format_instructions(req.post_format))

    if req.extra_instructions:
        sections.append(
            f"## YÊU CẦU BỔ SUNG CỦA NGƯỜI DÙNG\n{req.extra_instructions}"
        )

    sections.append(json_output_instructions(req.post_format))

    return "\n\n".join(sections)


def format_instructions(post_format: str) -> str:
    base = "## ĐỊNH DẠNG BÀI ĐĂNG"

    if post_format == "text":
        return (
            f"{base}: TEXT\n"
            "Viết bài thuần văn bản cho Facebook. Cấu trúc: hook mạnh → "
            "nội dung chính → call-to-action. Độ dài 150-300 từ."
        )

    if post_format == "image_text":
        return (
            f"{base}: IMAGE + TEXT\n"
            "Caption đi kèm ảnh. Hook ngắn gọn (1-2 câu đầu phải cuốn). "
            "Nội dung 80-200 từ. Kết thúc bằng CTA rõ ràng."
        )

    if post_format == "video_script":
        return (
            f"{base}: VIDEO SCRIPT\n"
            "Sinh cả caption đăng kèm video VÀ kịch bản quay chi tiết. "
            "Kịch bản chia thành: [HOOK 0-3s] → [NỘI DUNG CHÍNH] → [CTA]. "
            "Caption 50-100 từ. Kịch bản 200-400 từ."
        )

    return base


def json_output_instructions(post_format: str) -> str:
    video_field = (
        '\n    "video_script": "Kịch bản quay đầy đủ (phân cảnh theo thời gian)",'
        if post_format == "video_script"
        else '\n    "video_script": null,'
    )

    return (
        "## YÊU CẦU OUTPUT\n"
        "Chỉ trả về JSON hợp lệ theo cấu trúc sau, KHÔNG kèm markdown, "
        "KHÔNG giải thích thêm bên ngoài JSON:\n\n"
        "```json\n"
        "{\n"
        '    "caption": "Toàn bộ nội dung caption / bài viết",'
        f"{video_field}\n"
        '    "hook": "Câu mở đầu / hook (trích từ caption)",\n'
        '    "call_to_action": "Lời kêu gọi hành động cuối bài",\n'
        '    "hashtags": ["hashtag1", "hashtag2", "hashtag3"],\n'
        '    "ai_notes": "Giải thích ngắn tại sao chọn góc tiếp cận này"\n'
        "}\n"
        "```"
    )


def parse_response(raw: str, post_format: str) -> GeneratedContent:
    """Parse JSON response từ LLM, fallback về raw text nếu không hợp lệ."""
    json_str = extract_json(raw)

    if json_str:
        try:
            data = json.loads(json_str)

            return GeneratedContent(
                caption=data.get("caption", raw),
                video_script=data.get("video_script"),
                hook=data.get("hook")
                or extract_first_sentence(data.get("caption", raw)),
                call_to_action=data.get("call_to_action"),
                hashtags=data.get("hashtags") or [],
                ai_notes=data.get("ai_notes"),
            )
        except (json.JSONDecodeError, TypeError):
            pass

    return GeneratedContent(
        caption=raw,
        hook=extract_first_sentence(raw),
    )


def extract_json(text: str) -> str | None:
    """Tìm khối JSON đầu tiên trong text."""
    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        re.DOTALL,
    )

    if fence_match:
        return fence_match.group(1)

    start = text.find("{")

    if start == -1:
        return None

    depth = 0

    for i, char in enumerate(text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

            if depth == 0:
                return text[start : i + 1]

    return None


def extract_first_sentence(text: str) -> str:
    """Lấy câu đầu tiên làm hook fallback."""
    text = text.strip()

    for separator in (".", "!", "?", "\n"):
        index = text.find(separator)

        if index != -1 and index < 200:
            return text[: index + 1].strip()

    return text[:200].strip()
