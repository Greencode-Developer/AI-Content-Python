from ai_python.presentation.schemas.generate_idea import GenerateIdeaResponse


class GeminiAIClient:

    async def generate(self, prompt: str) -> str:
        return """
        {
            "title": "Khuyến mãi cà phê cuối tuần",
            "description": "Giới thiệu chương trình ưu đãi cà phê cuối tuần.",
            "hook": "Cuối tuần này, bạn đã có lý do để ghé quán chưa?"
        }
        """