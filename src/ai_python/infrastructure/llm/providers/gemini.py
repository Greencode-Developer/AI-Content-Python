from google import genai
from google.genai.errors import ServerError

from fastapi import HTTPException

_GEMINI_MODEL = "gemini-3.8-flash"

# Tên model hiển thị để ghi vào trường ai_model_used
MODEL_NAME = _GEMINI_MODEL


class GeminiAIClient:

    def __init__(self):
        self.client = genai.Client()

    async def generate(self, prompt: str) -> str:
        try:
            response = await self.client.aio.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt,
            )
            return response.text
        except ServerError as e:
            status = e.status_code if hasattr(e, "status_code") else 503
            raise HTTPException(
                status_code=status,
                detail=f"Gemini API error: {e}",
            ) from e