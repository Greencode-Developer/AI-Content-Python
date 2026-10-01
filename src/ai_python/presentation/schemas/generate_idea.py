from pydantic import BaseModel


class GenerateIdeaRequest(BaseModel):
    topic: str
    brand_name: str
    tone: str
    persona: str
    content_pillar: str


class Idea(BaseModel):
    title: str
    description: str
    hook: str


class GenerateIdeaResponse(BaseModel):
    ideas: list[Idea]