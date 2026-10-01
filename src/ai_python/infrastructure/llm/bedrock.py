from fastapi import Depends

from ai_python.service.idea.generate_idea import GenerateIdea


def get_generate_idea(
    llm: BedrockLLM = Depends(get_llm),
) -> GenerateIdea:
    return GenerateIdea(llm)
