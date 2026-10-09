from ai_python.schemas.generate_idea import GenerateIdeaRequest


def build_generate_idea_prompt(
    request: GenerateIdeaRequest,
) -> str:
    return f"""
You are a content strategist for small businesses.

## Task
Generate 3 distinct content ideas based on the provided
brand information and content requirements.

## Input
- Topic: {request.topic}
- Brand: {request.brand_name}
- Tone: {request.tone}
- Target persona: {request.persona}
- Content pillar: {request.content_pillar}

## Requirements
- Make each idea relevant to the topic.
- Match the brand's tone of voice.
- Address the target persona's needs and interests.
- Align every idea with the specified content pillar.
- Do not invent additional brand information or persona details.
- Avoid generic, repetitive, or overlapping ideas.
- Write each title, description, and hook in the same language
  as the input.

## Output format
Return a valid JSON object with exactly this structure:
{{
  "ideas": [
    {{
      "title": "A concise content idea title",
      "description": "A practical explanation of the idea",
      "hook": "An engaging opening sentence"
    }}
  ]
}}

## Output constraints
- Return exactly 3 ideas.
- Every idea must contain title, description, and hook.
- All values must be strings.
- Do not include additional fields.
- Return only the JSON object.
- Do not wrap the JSON in Markdown code fences.
""".strip()
