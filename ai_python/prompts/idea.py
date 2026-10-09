from ai_python.schemas.generate_idea import GenerateIdeaRequest

def build_generate_idea_prompt(
    request: GenerateIdeaRequest,
) -> str:
    return f"""
You are a content strategist for small businesses.

Your task is to generate content ideas based on the provided
brand information and content requirements.

Input:
- Topic: {request.topic}
- Brand: {request.brand_name}
- Tone: {request.tone}
- Target persona: {request.persona}
- Content pillar: {request.content_pillar}

Requirements:
- Generate relevant and practical content ideas.
- Match the brand's tone of voice.
- Address the needs and interests of the target persona.
- Align each idea with the specified content pillar.
- Do not invent brand information or persona details.
- Avoid generic and repetitive ideas.

Return the generated content ideas clearly.
""".strip()