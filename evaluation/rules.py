"""Rule-based checks for the raw output of the generate-ideas prompt.

Every rule comes from a requirement written in the prompt. The rules run on the
raw model output, before the service's retry, so a run shows how often the
model gets it right on the first try. Quality (relevance, creativity) is left
to the LLM judge.
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations

from pydantic import ValidationError

from app.core.llm_output import extract_json
from app.schemas.generate_idea import GenerateIdeaRequest, GenerateIdeaResponse, Idea

# Share of accented letters in a typical Vietnamese sentence is ~25%, English is ~0%
MIN_VIETNAMESE_RATIO = 0.1

# Titles more similar than this count as the same idea
MAX_TITLE_SIMILARITY = 0.8

IDEA_FIELDS = tuple(Idea.model_fields)

# Placeholders the model copies from the prompt instead of using the input values.
# Plain words like "topic" or "persona" are left out: Vietnamese marketing text uses them
PLACEHOLDER_PATTERN = re.compile(r"\{\{?\s*\w+\s*\}?\}|\bbrand_name\b|\bcontent_pillar\b")


@dataclass(frozen=True)
class RuleResult:
    name: str
    passed: bool
    detail: str = ""


def evaluate(request: GenerateIdeaRequest, raw_output: str) -> list[RuleResult]:
    data, parse_error = _parse(raw_output)

    results = [
        RuleResult("valid_json", data is not None, parse_error),
        _json_only(raw_output),
    ]

    if data is None:
        return results + _skipped("no_extra_fields", "schema_valid", "vietnamese_output",
                                  "no_placeholder", "distinct_ideas")

    results.append(_no_extra_fields(data))

    try:
        response = GenerateIdeaResponse.model_validate(data)
    except ValidationError as exc:
        results.append(RuleResult("schema_valid", False, _first_error(exc)))
        return results + _skipped("vietnamese_output", "no_placeholder", "distinct_ideas")

    results.append(RuleResult("schema_valid", True))
    results.append(_vietnamese_output(response))
    results.append(_no_placeholder(request, response))
    results.append(_distinct_ideas(response))
    return results


def score(results: list[RuleResult]) -> float:
    return sum(result.passed for result in results) / len(results) if results else 0.0


def parse_response(raw_output: str) -> GenerateIdeaResponse | None:
    """The ideas the service would return, or None when the output breaks the schema."""
    data, _ = _parse(raw_output)
    if data is None:
        return None
    try:
        return GenerateIdeaResponse.model_validate(data)
    except ValidationError:
        return None


def _parse(raw_output: str) -> tuple[dict | None, str]:
    try:
        data = json.loads(extract_json(raw_output) or raw_output)
    except json.JSONDecodeError as exc:
        # e.g. "Expecting ',' delimiter at line 4": the model often drops commas between fields
        return None, f"{exc.msg} at line {exc.lineno}"

    if not isinstance(data, dict):
        return None, "JSON is not an object"
    return data, ""


def _json_only(raw_output: str) -> RuleResult:
    # The prompt asks for bare JSON: no ```json fence, no text around it
    try:
        if isinstance(json.loads(raw_output), dict):
            return RuleResult("json_only", True)
    except json.JSONDecodeError:
        pass

    detail = "wrapped in a code fence" if "```" in raw_output else "not a bare JSON object"
    return RuleResult("json_only", False, detail)


def _no_extra_fields(data: dict) -> RuleResult:
    extra = set(data) - {"ideas"}

    for idea in data.get("ideas") or []:
        if isinstance(idea, dict):
            extra |= set(idea) - set(IDEA_FIELDS)

    return RuleResult("no_extra_fields", not extra, ", ".join(sorted(extra)))


def _vietnamese_output(response: GenerateIdeaResponse) -> RuleResult:
    for index, idea in enumerate(response.ideas):
        text = " ".join(getattr(idea, field) for field in IDEA_FIELDS)
        ratio = _accented_ratio(text)

        if ratio < MIN_VIETNAMESE_RATIO:
            return RuleResult("vietnamese_output", False, f"idea {index}: accented letters {ratio:.0%}")

    return RuleResult("vietnamese_output", True)


def _accented_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0

    # NFD splits "ế" into "e" + combining marks; "đ" has no decomposition
    accented = sum(
        1 for char in letters
        if char in "đĐ" or len(unicodedata.normalize("NFD", char)) > 1
    )
    return accented / len(letters)


def _no_placeholder(request: GenerateIdeaRequest, response: GenerateIdeaResponse) -> RuleResult:
    inputs = " ".join(request.model_dump().values())

    for index, idea in enumerate(response.ideas):
        for field in IDEA_FIELDS:
            for match in PLACEHOLDER_PATTERN.findall(getattr(idea, field)):
                # A brand like "{Mia}" is a real input value, not a placeholder
                if match not in inputs:
                    return RuleResult("no_placeholder", False, f"idea {index} {field}: {match}")

    return RuleResult("no_placeholder", True)


def _distinct_ideas(response: GenerateIdeaResponse) -> RuleResult:
    titles = [idea.title.casefold().strip() for idea in response.ideas]

    for (i, first), (j, second) in combinations(enumerate(titles), 2):
        similarity = SequenceMatcher(None, first, second).ratio()

        if similarity > MAX_TITLE_SIMILARITY:
            return RuleResult("distinct_ideas", False, f"ideas {i} and {j}: {similarity:.0%} similar")

    return RuleResult("distinct_ideas", True)


def _first_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error["loc"])
    return f"{location}: {error['msg']}"


def _skipped(*names: str) -> list[RuleResult]:
    return [RuleResult(name, False, "skipped: earlier check failed") for name in names]
