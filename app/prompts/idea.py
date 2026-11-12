from langfuse import get_client
from langfuse.model import TextPromptClient

from app.schemas.generate_idea import GenerateIdeaRequest

langfuse = get_client()


def get_generate_idea_prompt(label: str = "production") -> TextPromptClient:
    # The eval passes another label (e.g. "candidate") to test a version before it goes to production
    return langfuse.get_prompt(
        "generate-ideas",
        type="text",
        label=label,
    )


def compile_generate_idea_prompt(
    prompt: TextPromptClient,
    request: GenerateIdeaRequest,
) -> str:
    return prompt.compile(
        topic=request.topic,
        brand_name=request.brand_name,
        tone=request.tone,
        persona=request.persona,
        content_pillar=request.content_pillar,
    )


def build_generate_idea_prompt(
    request: GenerateIdeaRequest,
) -> str:
    return compile_generate_idea_prompt(get_generate_idea_prompt(), request)
