import json

from langfuse import get_client
from pydantic import ValidationError
from app.core.exception.exception import InvalidLLMOutputError
from app.core.llm_output import extract_json
from app.infrastructure.llm.providers.base import AIClient
from app.schemas.generate_idea import IDEA_JSON_SCHEMA, GeneratedIdeas, GenerateIdeaRequest, GenerateIdeaResponse
from app.prompts.idea import compile_generate_idea_prompt, get_generate_idea_prompt


MAX_RETRIES = 2

langfuse = get_client()


async def generate_idea_service(
    request: GenerateIdeaRequest,
    ai_client: AIClient,
) -> GenerateIdeaResponse:
    # Kept as an object, not only the text: its version goes into the response
    prompt = get_generate_idea_prompt()

    # One trace per request: the model calls of every retry are nested under this span
    with langfuse.start_as_current_observation(
        name="generate-ideas",
        input=request.model_dump(),
        metadata={"prompt_name": prompt.name, "prompt_version": prompt.version},
    ) as span:
        generated = await _generate_valid_ideas(compile_generate_idea_prompt(prompt, request), ai_client)
        span.update(output=generated.model_dump())
        # None when Langfuse is off or has no keys; the Java worker accepts a null trace id
        trace_id = langfuse.get_current_trace_id()

    return GenerateIdeaResponse(
        ideas=generated.ideas,
        trace_id=trace_id,
        model=ai_client.model,
        prompt_version=str(prompt.version),
    )


async def _generate_valid_ideas(prompt: str, ai_client: AIClient) -> GeneratedIdeas:
    for attempt in range(MAX_RETRIES + 1):
        result = await ai_client.generate(prompt, json_schema=IDEA_JSON_SCHEMA)

        try:
            # Models often wrap the JSON in a ```json fence
            data = json.loads(extract_json(result) or result)
            return GeneratedIdeas.model_validate(data)

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
