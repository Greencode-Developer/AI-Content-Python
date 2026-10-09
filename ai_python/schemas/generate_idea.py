from pydantic import BaseModel,Field


class GenerateIdeaRequest(BaseModel):
    topic: str
    brand_name: str
    tone: str
    persona: str
    content_pillar: str


class Idea(BaseModel):
    title: str = Field(min_length=1)
    approach_angle: str = Field(min_length=1)
    hook_sentence: str = Field(min_length=1)
    reason: str = Field(min_length=1)



class GenerateIdeaResponse(BaseModel):
    ideas: list[Idea]