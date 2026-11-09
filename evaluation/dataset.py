from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.schemas.generate_idea import GenerateIdeaRequest

DEFAULT_DATASET = Path(__file__).parent / "datasets" / "generate_ideas.jsonl"

Category = Literal["normal", "edge", "hard"]


class EvalCase(BaseModel):
    id: str
    category: Category
    # Same body the Java worker sends to /api/v1/ideas/generate
    input: GenerateIdeaRequest
    # What a reviewer (or the LLM judge) should look at for this case
    notes: str = ""


def load_dataset(path: Path = DEFAULT_DATASET) -> list[EvalCase]:
    cases = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(EvalCase.model_validate_json(line))

    return cases
