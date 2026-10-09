import json
from ai_python.llm.providers.base import AIClient
from ai_python.schemas.generate_idea import GenerateIdeaRequest, GenerateIdeaResponse, Idea


from ai_python.llm.providers.base import AIClient
from ai_python.prompts.idea import build_generate_idea_prompt

class GenerateIdea:

    def __init__(self, ai_client: AIClient):
        self.ai_client = ai_client

    async def execute(
        self,
        request: GenerateIdeaRequest,
    ) -> GenerateIdeaResponse:
        prompt = build_generate_idea_prompt(request)

        result = await self.ai_client.generate(prompt)
        data = json.loads(result)

        return GenerateIdeaResponse(
            ideas=[
                Idea(**idea)
                for idea in data["ideas"]
            ]
        )