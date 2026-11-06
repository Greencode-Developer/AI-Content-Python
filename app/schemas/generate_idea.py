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


class GenerateIdeaResponse(BaseModel):
    # The prompt asks for exactly IDEA_COUNT ideas, so any other count is retried
    ideas: list[Idea] = Field(min_length=IDEA_COUNT, max_length=IDEA_COUNT)
