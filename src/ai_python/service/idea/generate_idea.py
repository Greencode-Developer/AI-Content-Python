from ai_python.presentation.schemas.generate_idea import GenerateIdeaRequest, GenerateIdeaResponse


class GenerateIdea:

    def __init__(self, llm):
        self.llm = llm

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

        ideas = await self.llm.generate(prompt)

        return GenerateIdeaResponse(
            ideas=ideas
        )
