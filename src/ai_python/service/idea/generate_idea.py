from ai_python.infrastructure.llm.base import AIClient
from ai_python.presentation.schemas.generate_idea import GenerateIdeaRequest, GenerateIdeaResponse, Idea

class GenerateIdea:

    def __init__(self, ai_client: AIClient):
        self.ai_client = ai_client

    async def execute(
        self,
        request: GenerateIdeaRequest,
    ) -> GenerateIdeaResponse:

        prompt = f"""
        Generate content ideas for:

        Topic: {request.topic}
        Brand: {request.brand_name}
        Tone: {request.tone}
        Persona: {request.persona}
        Content pillar: {request.content_pillar}
        """

        result = await self.ai_client.generate(prompt)

        return GenerateIdeaResponse(
            ideas=[Idea(
                title="Test title",
                description=result,
                hook="Test hook",
            )]
        )