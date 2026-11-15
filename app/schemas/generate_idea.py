from pydantic import BaseModel, Field

IDEA_COUNT = 3

# Same limit as ideas.title in the backend DB
TITLE_MAX_LENGTH = 255


class GenerateIdeaRequest(BaseModel):
    topic: str
    brand_name: str
    tone: str
    persona: str
    content_pillar: str


class Idea(BaseModel):
    title: str = Field(min_length=1, max_length=TITLE_MAX_LENGTH)
    approach_angle: str = Field(min_length=1)
    hook_sentence: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class GeneratedIdeas(BaseModel):
    """The part of the response the model writes."""

    # The prompt asks for exactly IDEA_COUNT ideas, so any other count is retried
    ideas: list[Idea] = Field(min_length=IDEA_COUNT, max_length=IDEA_COUNT)


class GenerateIdeaResponse(GeneratedIdeas):
    # Filled by the service, not the model. The Java worker saves them on the AI job
    # (ai_jobs.trace_id, model, prompt_version) to find the trace of a bad idea later
    trace_id: str | None = Field(default=None, description="Langfuse trace id, None when tracing is off")
    model: str | None = Field(default=None, description="Model id the provider called")
    prompt_version: str | None = Field(default=None, description="Version of the generate-ideas prompt in Langfuse")


# Passed to the provider so the answer is always valid JSON. Idea is inlined:
# a $ref to $defs (what Pydantic generates for the full response) is not supported by every provider
IDEA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "ideas": {
            "type": "array",
            "minItems": IDEA_COUNT,
            "maxItems": IDEA_COUNT,
            "items": Idea.model_json_schema(),
        },
    },
    "required": ["ideas"],
}
