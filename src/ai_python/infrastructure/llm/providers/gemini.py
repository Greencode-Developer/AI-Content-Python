from ai_python.presentation.schemas.generate_idea import GenerateIdeaResponse
from google import genai


class GeminiAIClient:

    def __init__(self):
        self.client = genai.Client()

    async def generate(self, prompt: str) -> str:
        response = await self.client.aio.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
        )

        return response.text