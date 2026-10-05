"""
Service sinh bài đăng Facebook.

Luồng:
  1. Nhận GenerateContentRequest (brand context, persona, idea, format)
  2. Xây dựng prompt yêu cầu AI trả về JSON thuần
  3. Gọi AIClient.generate()
  4. Parse JSON response → GenerateContentResponse
  5. Nếu parse thất bại → fallback giữ nguyên raw text trong caption
"""

import json
import re

from ai_python.infrastructure.llm.base import AIClient
from ai_python.presentation.schemas.generate_content import (
    GenerateContentRequest,
    GenerateContentResponse,
    GeneratedContent,
)

# Model name được lấy từ GeminiAIClient — đồng bộ để ghi vào ai_model_used
_DEFAULT_MODEL_NAME = "gemini-3.8-flash"


class GenerateContent:

    def __init__(self, ai_client: AIClient):
        self.ai_client = ai_client

    # ── Public ──────────────────────────────────────────────────────────────

    async def execute(
        self,
        request: GenerateContentRequest,
    ) -> GenerateContentResponse:
        prompt = self._build_prompt(request)
        raw = await self.ai_client.generate(prompt)
        content = self._parse_response(raw, request.post_format)

        return GenerateContentResponse(
            content=content,
            ai_model_used=_DEFAULT_MODEL_NAME,
            idea_id=request.idea_id,
            pillar_id=request.pillar_id,
            persona_id=request.persona_id,
            channel_id=request.channel_id,
        )

    # ── Prompt builder ───────────────────────────────────────────────────────

    def _build_prompt(self, req: GenerateContentRequest) -> str:
        sections: list[str] = []

        # 1. Vai trò & nhiệm vụ
        sections.append(
            "Bạn là chuyên gia viết nội dung mạng xã hội cho các shop nhỏ tại Việt Nam. "
            "Nhiệm vụ: sinh một bài đăng Facebook hoàn chỉnh theo thông tin dưới đây."
        )

        # 2. Thông tin thương hiệu
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

        # 3. Chân dung khách hàng
        if req.persona:
            p = req.persona
            persona_lines = [
                "## CHÂN DUNG KHÁCH HÀNG MỤC TIÊU",
                f"- Tên chân dung: {p.name}",
            ]
            if p.age_range:
                persona_lines.append(f"- Độ tuổi: {p.age_range}")
            if p.occupation:
                persona_lines.append(f"- Nghề nghiệp: {p.occupation}")
            if p.pain_points:
                persona_lines.append(f"- Nỗi đau / vấn đề: {p.pain_points}")
            if p.desires:
                persona_lines.append(f"- Mong muốn: {p.desires}")
            if p.typical_phrases:
                persona_lines.append(f"- Câu họ hay nói: {p.typical_phrases}")
            sections.append("\n".join(persona_lines))

        # 4. Trụ cột nội dung
        if req.pillar_name:
            sections.append(
                f"## TRỤ CỘT NỘI DUNG\n- Trụ cột: {req.pillar_name}\n"
                "  (Bài viết phải phục vụ đúng mục tiêu của trụ cột này)"
            )

        # 5. Ý tưởng nguồn
        if req.idea:
            idea = req.idea
            idea_lines = [
                "## Ý TƯỞNG NGUỒN",
                f"- Tiêu đề: {idea.title}",
            ]
            if idea.approach_angle:
                idea_lines.append(f"- Góc tiếp cận: {idea.approach_angle}")
            if idea.hook_sentence:
                idea_lines.append(f"- Câu hook gợi ý: {idea.hook_sentence}")
            sections.append("\n".join(idea_lines))

        # 6. Định dạng bài
        format_guide = self._format_instructions(req.post_format)
        sections.append(format_guide)

        # 7. Hướng dẫn bổ sung từ người dùng
        if req.extra_instructions:
            sections.append(
                f"## YÊU CẦU BỔ SUNG CỦA NGƯỜI DÙNG\n{req.extra_instructions}"
            )

        # 8. Cấu trúc JSON output bắt buộc
        sections.append(self._json_output_instructions(req.post_format))

        return "\n\n".join(sections)

    # ── Instruction helpers ──────────────────────────────────────────────────

    @staticmethod
    def _format_instructions(post_format: str) -> str:
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

    @staticmethod
    def _json_output_instructions(post_format: str) -> str:
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

    # ── Response parser ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(raw: str, post_format: str) -> GeneratedContent:
        """
        Trích JSON từ response của LLM.
        LLM đôi khi bọc JSON trong ```json ... ``` hoặc thêm text thừa.
        Fallback: nếu không parse được thì gói raw vào caption.
        """
        json_str = _extract_json(raw)
        if json_str:
            try:
                data = json.loads(json_str)
                return GeneratedContent(
                    caption=data.get("caption", raw),
                    video_script=data.get("video_script"),
                    hook=data.get("hook") or _extract_first_sentence(data.get("caption", raw)),
                    call_to_action=data.get("call_to_action"),
                    hashtags=data.get("hashtags") or [],
                    ai_notes=data.get("ai_notes"),
                )
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback: trả về raw text, cố gắng lấy hook từ câu đầu
        return GeneratedContent(
            caption=raw,
            hook=_extract_first_sentence(raw),
        )


# ── Module-level helpers ─────────────────────────────────────────────────────

def _extract_json(text: str) -> str | None:
    """Tìm khối JSON đầu tiên trong text (có thể bọc trong ```json ... ```)."""
    # Thử bóc từ code fence trước
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    # Không có fence — tìm { ... } outermost
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_first_sentence(text: str) -> str:
    """Lấy câu đầu tiên làm hook fallback."""
    text = text.strip()
    for sep in (".", "!", "?", "\n"):
        idx = text.find(sep)
        if idx != -1 and idx < 200:
            return text[: idx + 1].strip()
    return text[:200].strip()
