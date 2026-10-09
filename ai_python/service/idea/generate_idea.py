import json

from pydantic import ValidationError
from ai_python.llm.providers.base import AIClient
from ai_python.schemas.generate_idea import GenerateIdeaRequest, GenerateIdeaResponse
from ai_python.llm.providers.base import AIClient
from ai_python.prompts.idea import build_generate_idea_prompt

class InvalidLLMOutputError(Exception):
    pass

class GenerateIdea:
    MAX_RETRIES = 2

    def __init__(self, ai_client: AIClient):
        self.ai_client = ai_client

    async def execute(
        self,
        request: GenerateIdeaRequest,
    ) -> GenerateIdeaResponse:
        prompt = build_generate_idea_prompt(request)

        for attempt in range(self.MAX_RETRIES + 1):
            result = await self.ai_client.generate(prompt)

            try:
                data = json.loads(result)
                return GenerateIdeaResponse.model_validate(data)

            except (json.JSONDecodeError, ValidationError) as exc:
                if attempt == self.MAX_RETRIES:
                    raise InvalidLLMOutputError(
                        "LLM returned an invalid idea response"
                    ) from exc

                prompt = (
                    f"{prompt}\n\n"
                    "Your previous response did not satisfy the required "
                    "JSON format or schema. Generate a corrected response. "
                    "Return only valid JSON matching the required schema."
                )

        raise InvalidLLMOutputError("Failed to generate valid ideas")