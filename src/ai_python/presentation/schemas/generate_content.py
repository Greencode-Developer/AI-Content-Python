"""
Schemas cho API sinh bài đăng (POST /content/generate).

Dữ liệu đầu vào bám theo ERD: contents, ideas, brand_profiles,
content_pillars, personas. AI trả về structured JSON gồm caption
(và video_script nếu post_format là video_script).
"""

from typing import Literal
from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────

PostFormat = Literal["text", "image_text", "video_script"]


# ── Sub-models cho context đầu vào ─────────────────────────────────────────

class BrandContext(BaseModel):
    """Thông tin thương hiệu truyền vào prompt."""

    description: str = Field(..., description="Mô tả thương hiệu / sản phẩm")
    tone_of_voice: str = Field(..., description="Giọng điệu (ấm áp, chuyên nghiệp, hài hước...)")
    forbidden_words: str | None = Field(
        default=None, description="Từ / cụm từ cấm — cách nhau bằng dấu phẩy"
    )


class PersonaContext(BaseModel):
    """Chân dung khách hàng nhắm tới."""

    name: str = Field(..., description="Tên chân dung (VD: Mẹ bỉm sữa 28-35)")
    age_range: str | None = Field(default=None)
    occupation: str | None = Field(default=None)
    pain_points: str | None = Field(default=None, description="Nỗi đau / vấn đề")
    desires: str | None = Field(default=None, description="Mong muốn / mục tiêu")
    typical_phrases: str | None = Field(default=None, description="Câu họ hay nói")


class IdeaContext(BaseModel):
    """Ý tưởng nguồn — lấy từ bảng ideas trong DB."""

    title: str = Field(..., description="Tiêu đề ý tưởng")
    approach_angle: str | None = Field(default=None, description="Góc tiếp cận")
    hook_sentence: str | None = Field(default=None, description="Câu mở đầu gợi ý")


# ── Request ────────────────────────────────────────────────────────────────

class GenerateContentRequest(BaseModel):
    """
    Payload để sinh một bài đăng Facebook.

    Các trường `*_id` (pillar_id, persona_id, idea_id, channel_id) là
    tham chiếu tới DB — service dùng để lưu kết quả vào bảng contents.
    Các trường context (`brand`, `persona`, `idea`) là dữ liệu thực tế
    được truyền vào prompt AI để không cần service gọi lại DB.
    """

    # ── Định dạng & kênh ──
    post_format: PostFormat = Field(
        default="text",
        description="Định dạng bài: text | image_text | video_script",
    )
    platform: Literal["fanpage"] = Field(default="fanpage")

    # ── Tham chiếu DB (nullable, dùng để gắn FK khi lưu) ──
    idea_id: str | None = Field(default=None, description="UUID của idea nguồn")
    pillar_id: str | None = Field(default=None, description="UUID của content_pillar")
    persona_id: str | None = Field(default=None, description="UUID của persona")
    channel_id: str | None = Field(default=None, description="UUID của channel dự kiến đăng")

    # ── Context truyền vào AI ──
    brand: BrandContext = Field(..., description="Thông tin thương hiệu")
    persona: PersonaContext | None = Field(
        default=None, description="Chân dung khách hàng (nếu có)"
    )
    idea: IdeaContext | None = Field(
        default=None, description="Ý tưởng nguồn (nếu sinh từ idea)"
    )
    pillar_name: str | None = Field(
        default=None, description="Tên trụ cột nội dung (VD: Giáo dục, Bán hàng)"
    )

    # ── Ghi chú thêm từ người dùng ──
    extra_instructions: str | None = Field(
        default=None,
        description="Hướng dẫn bổ sung từ người dùng (VD: tập trung vào giá ưu đãi)",
    )


# ── Response ───────────────────────────────────────────────────────────────

class GeneratedContent(BaseModel):
    """Nội dung bài đăng AI trả về."""

    caption: str = Field(..., description="Caption / nội dung bài viết hoàn chỉnh")
    video_script: str | None = Field(
        default=None,
        description="Kịch bản quay (chỉ có khi post_format=video_script)",
    )
    hook: str = Field(..., description="Câu mở đầu / hook của bài")
    call_to_action: str | None = Field(
        default=None, description="Lời kêu gọi hành động cuối bài"
    )
    hashtags: list[str] = Field(
        default_factory=list,
        description="Danh sách hashtag gợi ý (không có dấu #)",
    )
    ai_notes: str | None = Field(
        default=None,
        description="Ghi chú của AI về lý do chọn góc tiếp cận này",
    )


class GenerateContentResponse(BaseModel):
    """Response của POST /content/generate."""

    content: GeneratedContent
    ai_model_used: str = Field(..., description="Tên model AI đã sinh nội dung")
    # Các trường này được echo lại để FE tiện lưu vào DB
    idea_id: str | None = None
    pillar_id: str | None = None
    persona_id: str | None = None
    channel_id: str | None = None
