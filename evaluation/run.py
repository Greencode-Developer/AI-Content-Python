"""Run the eval dataset through the generate-ideas prompt and score each output.

    uv run python -m evaluation.run --provider bedrock --limit 5
    uv run python -m evaluation.run --provider bedrock --judge
    uv run python -m evaluation.run --from-results evaluation/results/<file>.json --judge
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
from evaluation.judge import DEFAULT_JUDGE, JUDGE_MODELS, SCORED_CRITERIA, JudgeVerdict, create_judge_client, judge
from evaluation.rules import RuleResult, evaluate, parse_response, score

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
    # Only set when the run uses a judge and the output passes the schema
    judge: dict | None = None
    judge_error: str | None = None


async def run_experiment(
    cases: list[EvalCase],
    ai_client: AIClient | None,
    build_prompt: Callable[[GenerateIdeaRequest], str] = build_generate_idea_prompt,
    judge_client: AIClient | None = None,
    replay: dict[str, CaseResult] | None = None,
) -> list[CaseResult]:
    """Generate (or replay saved outputs), run the rules, then the judge when one is given."""
    results = []

    # One case at a time: provider rate limits are low and the dataset is small
    for case in cases:
        result = CaseResult(id=case.id, category=case.category)

        if replay is not None:
            # Re-score saved outputs: no generation call, so judges can be compared on the same output
            result.raw_output, result.error = replay[case.id].raw_output, replay[case.id].error
        else:
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

        response = parse_response(result.raw_output) if result.error is None else None
        if judge_client is not None and response is not None:
            try:
                result.judge = (await judge(case, response, judge_client)).model_dump()
            except Exception as exc:
                result.judge_error = f"{type(exc).__name__}: {exc}"

        results.append(result)
        print(f"  {case.id}: {_progress(result)}")

    return results


def _progress(result: CaseResult) -> str:
    if result.error:
        return "ERROR"
    line = f"rules {result.score:.0%}"
    if result.judge:
        line += f", judge {JudgeVerdict.model_validate(result.judge).average():.1f}/5"
    elif result.judge_error:
        line += ", judge ERROR"
    return line


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

    lines.extend(_summarize_judge(scored))
    return "\n".join(lines)


def _summarize_judge(scored: list[CaseResult]) -> list[str]:
    judged = [(result, JudgeVerdict.model_validate(result.judge)) for result in scored if result.judge]
    judge_errors = [result for result in scored if result.judge_error]
    if not judged and not judge_errors:
        return []

    lines = [f"\nJudge: {len(judged)} judged, {len(judge_errors)} judge errors"]
    if judged:
        lines.append(f"  {'average':<20} {_average(verdict.average() for _, verdict in judged):>5.1f}/5")
        for name in SCORED_CRITERIA:
            lines.append(f"  {name:<20} {_average(getattr(verdict, name).score for _, verdict in judged):>5.1f}/5")
        for label, attribute in (("invented facts", "invented_facts"), ("safety issues", "safety_issues")):
            flagged = [(result, getattr(verdict, attribute)) for result, verdict in judged if getattr(verdict, attribute)]
            lines.append(f"  {label:<20} {len(flagged)}/{len(judged)} outputs")
            lines.extend(f"    {result.id}: {'; '.join(items)}" for result, items in flagged)

    lines.extend(f"  {result.id}: {result.judge_error}" for result in judge_errors)
    return lines


def save_results(
    results: list[CaseResult],
    provider: str,
    model: str,
    judge_model: str | None = None,
    source: str | None = None,
    output_dir: Path = RESULTS_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now()
    suffix = f"-judge-{judge_model}" if judge_model else ""
    path = output_dir / f"{created_at:%Y%m%d-%H%M%S}-{provider}{suffix}.json"

    report = {
        "provider": provider,
        "model": model,
        "judge_model": judge_model,
        # The saved run whose outputs were re-scored, when --from-results is used
        "source": source,
        "created_at": created_at.isoformat(timespec="seconds"),
        "cases": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_results(path: Path) -> tuple[dict, dict[str, CaseResult]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    cases = {
        case["id"]: CaseResult(id=case["id"], category=case["category"], raw_output=case["raw_output"], error=case["error"])
        for case in report["cases"]
    }
    return report, cases


def _average(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the generate-ideas prompt")
    parser.add_argument("--provider", choices=[p.value for p in AIProvider], default=AIProvider.BEDROCK.value)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--category", choices=["normal", "edge", "hard"], help="only run this category")
    parser.add_argument("--limit", type=int, help="only run the first N cases")
    parser.add_argument("--judge", action="store_true", help="also score each output with the LLM judge")
    parser.add_argument("--judge-model", choices=list(JUDGE_MODELS), default=DEFAULT_JUDGE)
    parser.add_argument("--from-results", type=Path, help="re-score the outputs of a saved run instead of generating")
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    replay = None
    if args.from_results:
        source, replay = load_results(args.from_results)
        cases = [case for case in cases if case.id in replay]
        provider, model, ai_client = source["provider"], source["model"], None
    else:
        provider_enum = AIProvider(args.provider)
        provider, model, ai_client = provider_enum.value, MODEL_NAMES[provider_enum], get_ai_client(provider_enum)

    if args.category:
        cases = [case for case in cases if case.category == args.category]
    if args.limit:
        cases = cases[: args.limit]

    judge_model = args.judge_model if args.judge else None
    judge_client = create_judge_client(judge_model) if judge_model else None

    action = f"Re-scoring {args.from_results.name}" if args.from_results else "Running"
    print(f"{action}: {len(cases)} cases with {provider} ({model}), judge: {judge_model or 'off'}")
    results = asyncio.run(run_experiment(cases, ai_client, judge_client=judge_client, replay=replay))

    print()
    print(summarize(results))
    source_name = args.from_results.name if args.from_results else None
    print(f"\nSaved to {save_results(results, provider, model, judge_model, source_name)}")


if __name__ == "__main__":
    main()
