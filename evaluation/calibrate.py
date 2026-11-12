"""Check a judge against hand labels before trusting its scores.

Fill the "human" block of every item in datasets/judge_calibration.json, then:

    uv run python -m evaluation.calibrate                          # default judge
    uv run python -m evaluation.calibrate --judge-model haiku-4.5
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field

from app.infrastructure.llm.providers.base import AIClient
from app.schemas.generate_idea import GenerateIdeaResponse
from evaluation.dataset import load_dataset
from evaluation.judge import DEFAULT_JUDGE, JUDGE_MODELS, SCORED_CRITERIA, JudgeVerdict, create_judge_client, judge
from evaluation.run import RESULTS_DIR

CALIBRATION_FILE = Path(__file__).parent / "datasets" / "judge_calibration.json"

FLAGS = ("invented_facts", "safety_issues")


class HumanLabel(BaseModel):
    # true = the output has at least one invented fact / safety issue
    invented_facts: bool
    safety_issues: bool
    naturalness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    specificity: int = Field(ge=1, le=5)
    persona_fit: int = Field(ge=1, le=5)


class CalibrationItem(BaseModel):
    case_id: str
    source: str
    human: HumanLabel
    output: GenerateIdeaResponse


def load_calibration(path: Path = CALIBRATION_FILE) -> list[CalibrationItem]:
    raw_items = json.loads(path.read_text(encoding="utf-8"))

    missing = [
        f"  {item['case_id']}: {', '.join(name for name, value in item['human'].items() if value is None)}"
        for item in raw_items
        if None in item["human"].values()
    ]
    if missing:
        raise ValueError(f"Fill the human labels in {path.name} first. Missing:\n" + "\n".join(missing))

    return [CalibrationItem.model_validate(item) for item in raw_items]


async def judge_items(items: list[CalibrationItem], client: AIClient) -> list[JudgeVerdict | str]:
    cases = {case.id: case for case in load_dataset()}
    verdicts = []

    for item in items:
        try:
            verdicts.append(await judge(cases[item.case_id], item.output, client))
        except Exception as exc:
            verdicts.append(f"{type(exc).__name__}: {exc}")
        print(f"  {item.case_id}: {'ERROR' if isinstance(verdicts[-1], str) else 'judged'}")

    return verdicts


def compare(items: list[CalibrationItem], verdicts: list[JudgeVerdict | str]) -> str:
    """Per item human/judge scores, then how often the judge agrees with the human."""
    header = f"{'case':<24}" + "".join(f"{name[:11]:>13}" for name in SCORED_CRITERIA + FLAGS)
    lines = ["Human / judge", header]

    pairs = [(item.human, verdict) for item, verdict in zip(items, verdicts) if isinstance(verdict, JudgeVerdict)]

    for item, verdict in zip(items, verdicts):
        if isinstance(verdict, str):
            lines.append(f"{item.case_id:<24} ERROR {verdict}")
            continue
        cells = [f"{getattr(item.human, name)} / {getattr(verdict, name).score}" for name in SCORED_CRITERIA]
        cells += [f"{_yes(getattr(item.human, name))} / {_yes(bool(getattr(verdict, name)))}" for name in FLAGS]
        lines.append(f"{item.case_id:<24}" + "".join(f"{cell:>13}" for cell in cells))

    if not pairs:
        return "\n".join(lines)

    lines.append(f"\nAgreement on {len(pairs)} items:")
    for name in SCORED_CRITERIA:
        diffs = [abs(getattr(human, name) - getattr(verdict, name).score) for human, verdict in pairs]
        within_one = sum(diff <= 1 for diff in diffs)
        lines.append(
            f"  {name:<16} mean |human - judge| {sum(diffs) / len(diffs):.1f}, within 1 point {within_one}/{len(pairs)}"
        )
    for name in FLAGS:
        same = sum(getattr(human, name) == bool(getattr(verdict, name)) for human, verdict in pairs)
        missed = sum(getattr(human, name) and not getattr(verdict, name) for human, verdict in pairs)
        lines.append(f"  {name:<16} same answer {same}/{len(pairs)}, missed by judge {missed}")

    return "\n".join(lines)


def save_calibration(items: list[CalibrationItem], verdicts: list[JudgeVerdict | str], judge_model: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now()
    path = RESULTS_DIR / f"{created_at:%Y%m%d-%H%M%S}-calibration-{judge_model}.json"
    report = {
        "judge_model": judge_model,
        "created_at": created_at.isoformat(timespec="seconds"),
        "items": [
            {
                "case_id": item.case_id,
                "human": item.human.model_dump(),
                "judge": verdict.model_dump() if isinstance(verdict, JudgeVerdict) else None,
                "judge_error": verdict if isinstance(verdict, str) else None,
            }
            for item, verdict in zip(items, verdicts)
        ],
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the LLM judge with hand labels")
    parser.add_argument("--judge-model", choices=list(JUDGE_MODELS), default=DEFAULT_JUDGE)
    args = parser.parse_args()

    try:
        items = load_calibration()
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    print(f"Judging {len(items)} labelled outputs with {args.judge_model}")
    verdicts = asyncio.run(judge_items(items, create_judge_client(args.judge_model)))

    print()
    print(compare(items, verdicts))
    print(f"\nSaved to {save_calibration(items, verdicts, args.judge_model)}")


if __name__ == "__main__":
    main()
