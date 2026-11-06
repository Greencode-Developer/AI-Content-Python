import json

from pydantic import ValidationError
from app.core.exception.exception import InvalidLLMOutputError
from app.core.llm_output import extract_json
from app.infrastructure.llm.providers.base import AIClient
from app.schemas.generate_idea import GenerateIdeaRequest, GenerateIdeaResponse
from app.prompts.idea import build_generate_idea_prompt


MAX_RETRIES = 2

async def generate_idea_service(
    request: GenerateIdeaRequest,
    ai_client: AIClient,
) -> GenerateIdeaResponse:
    prompt = build_generate_idea_prompt(request)

    for attempt in range(MAX_RETRIES + 1):
        result = await ai_client.generate(prompt)

        try:
            # Models often wrap the JSON in a ```json fence
            data = json.loads(extract_json(result) or result)
            return GenerateIdeaResponse.model_validate(data)

        except (json.JSONDecodeError, ValidationError) as exc:
            if attempt == MAX_RETRIES:
                raise InvalidLLMOutputError(
                    "LLM returned an invalid idea response"
                ) from exc

            prompt += (
                "\n\nYour previous response did not satisfy the required "
                "JSON format or schema. Generate a corrected response. "
                "Return only valid JSON matching the required schema."
            )

    raise InvalidLLMOutputError("Failed to generate valid ideas")