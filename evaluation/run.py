"""Run the eval dataset through the generate-ideas prompt and score each output.

    uv run python -m evaluation.run --provider gemini --limit 5
"""

import argparse
import asyncio
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.infrastructure.llm.llm import AIProvider, get_ai_client
from app.infrastructure.llm.providers import bedrock, gemini
from app.infrastructure.llm.providers.base import AIClient
from app.prompts.idea import build_generate_idea_prompt
from app.schemas.generate_idea import GenerateIdeaRequest
from evaluation.dataset import DEFAULT_DATASET, EvalCase, load_dataset
from evaluation.rules import RuleResult, evaluate, score

RESULTS_DIR = Path(__file__).parent / "results"

# Saved with each run so results from different models can be compared later
MODEL_NAMES = {AIProvider.GEMINI: gemini.MODEL_NAME, AIProvider.BEDROCK: bedrock.MODEL_NAME}


@dataclass
class CaseResult:
    id: str
    category: str
    raw_output: str | None = None
    # Set when the provider call fails (throttling, timeout, no text...); the case has no rule results
    error: str | None = None
    rules: list[RuleResult] = field(default_factory=list)
    score: float = 0.0


async def run_experiment(
    cases: list[EvalCase],
    ai_client: AIClient,
    build_prompt: Callable[[GenerateIdeaRequest], str] = build_generate_idea_prompt,
) -> list[CaseResult]:
    results = []

    # One case at a time: provider rate limits are low and the dataset is small
    for case in cases:
        result = CaseResult(id=case.id, category=case.category)

        try:
            # One call, no retry: the service's retry would hide first-try failures
            result.raw_output = await ai_client.generate(build_prompt(case.input))
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"

        if result.raw_output is None and result.error is None:
            # Gemini returns None instead of text when a safety filter blocks the answer
            result.error = "provider returned no text"

        if result.error is None:
            result.rules = evaluate(case.input, result.raw_output)
            result.score = score(result.rules)

        results.append(result)
        print(f"  {case.id}: {'ERROR' if result.error else f'{result.score:.0%}'}")

    return results


def summarize(results: list[CaseResult]) -> str:
    scored = [result for result in results if result.error is None]
    lines = [f"Cases: {len(results)}, scored: {len(scored)}, errors: {len(results) - len(scored)}"]

    if not scored:
        return "\n".join(lines)

    lines.append(f"Average score: {_average(result.score for result in scored):.0%}")

    lines.append("\nPass rate by rule:")
    by_rule = defaultdict(list)
    for result in scored:
        for rule in result.rules:
            by_rule[rule.name].append(rule.passed)
    for name, passed in by_rule.items():
        lines.append(f"  {name:<20} {_average(passed):>5.0%}")

    lines.append("\nAverage score by category:")
    by_category = defaultdict(list)
    for result in scored:
        by_category[result.category].append(result.score)
    for category, scores in by_category.items():
        lines.append(f"  {category:<20} {_average(scores):>5.0%}")

    failures = [
        f"  {result.id}: {rule.name} ({rule.detail})" if rule.detail else f"  {result.id}: {rule.name}"
        for result in scored
        for rule in result.rules
        if not rule.passed and not rule.detail.startswith("skipped")
    ]
    if failures:
        lines.append("\nFailed rules:")
        lines.extend(failures)

    errors = [f"  {result.id}: {result.error}" for result in results if result.error]
    if errors:
        lines.append("\nErrors:")
        lines.extend(errors)

    return "\n".join(lines)


def save_results(results: list[CaseResult], provider: AIProvider, output_dir: Path = RESULTS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now()
    path = output_dir / f"{created_at:%Y%m%d-%H%M%S}-{provider.value}.json"

    report = {
        "provider": provider.value,
        "model": MODEL_NAMES[provider],
        "created_at": created_at.isoformat(timespec="seconds"),
        "cases": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _average(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the generate-ideas prompt")
    parser.add_argument("--provider", choices=[p.value for p in AIProvider], default=AIProvider.GEMINI.value)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--category", choices=["normal", "edge", "hard"], help="only run this category")
    parser.add_argument("--limit", type=int, help="only run the first N cases")
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    if args.category:
        cases = [case for case in cases if case.category == args.category]
    if args.limit:
        cases = cases[: args.limit]

    provider = AIProvider(args.provider)
    print(f"Running {len(cases)} cases with {provider.value} ({MODEL_NAMES[provider]})")
    results = asyncio.run(run_experiment(cases, get_ai_client(provider)))

    print()
    print(summarize(results))
    print(f"\nSaved to {save_results(results, provider)}")


if __name__ == "__main__":
    main()
