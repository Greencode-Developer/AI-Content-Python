from langfuse import get_client
from ai_python.schemas.generate_idea import GenerateIdeaRequest

langfuse = get_client()

def build_generate_idea_prompt(
    request: GenerateIdeaRequest,
) -> str:
    prompt = langfuse.get_prompt(
        "generate-ideas",
        type="text",
        label="production",
    )

    return prompt.compile(
        topic=request.topic,
        brand_name=request.brand_name,
        tone=request.tone,
        persona=request.persona,
        content_pillar=request.content_pillar,
    )