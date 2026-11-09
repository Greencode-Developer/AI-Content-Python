import json

import pytest
from pydantic import ValidationError

from app.schemas.generate_idea import GenerateIdeaResponse
from evaluation.calibrate import CALIBRATION_FILE, CalibrationItem, compare, load_calibration
from evaluation.dataset import load_dataset
from evaluation.judge import JudgeVerdict

OUTPUT = {"ideas": [{"title": f"Ý {i}", "approach_angle": "Góc", "hook_sentence": "Hook", "reason": "Lý do"} for i in range(3)]}


def item(**human) -> CalibrationItem:
    labels = {"invented_facts": True, "safety_issues": False, "naturalness": 2, "relevance": 4, "specificity": 3, "persona_fit": 3}
    return CalibrationItem.model_validate(
        {"case_id": "cafe-01", "source": "run.json", "human": {**labels, **human}, "output": OUTPUT}
    )


def verdict(naturalness: int, invented_facts: list[str]) -> JudgeVerdict:
    criterion = {"issues": [], "score": 3, "reason": "lý do"}
    return JudgeVerdict.model_validate(
        {
            "invented_facts": invented_facts,
            "safety_issues": [],
            "naturalness": {**criterion, "score": naturalness},
            "relevance": {**criterion, "score": 4},
            "specificity": criterion,
            "persona_fit": criterion,
        }
    )


def test_calibration_file_matches_the_dataset_and_the_output_schema():
    items = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    dataset_ids = {case.id for case in load_dataset()}

    assert len(items) == 10
    assert "cafe-01" in {item["case_id"] for item in items}
    for entry in items:
        assert entry["case_id"] in dataset_ids
        GenerateIdeaResponse.model_validate(entry["output"])


def test_unlabelled_item_is_reported_by_name(tmp_path):
    path = tmp_path / "calibration.json"
    labels = {**item().human.model_dump(), "naturalness": None, "persona_fit": None}
    path.write_text(json.dumps([{**item().model_dump(), "human": labels}]), encoding="utf-8")

    with pytest.raises(ValueError, match="cafe-01: naturalness, persona_fit"):
        load_calibration(path)


def test_label_outside_the_scale_is_rejected(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps([{**item().model_dump(), "human": {**item().human.model_dump(), "naturalness": 6}}]))

    with pytest.raises(ValidationError):
        load_calibration(path)


def test_compare_reports_score_gaps_and_missed_flags():
    report = compare([item(), item()], [verdict(naturalness=4, invented_facts=[]), verdict(naturalness=2, invented_facts=["x"])])

    assert "naturalness      mean |human - judge| 1.0, within 1 point 1/2" in report
    assert "invented_facts   same answer 1/2, missed by judge 1" in report


def test_judge_error_is_shown_and_left_out_of_the_agreement():
    report = compare([item(), item()], [verdict(naturalness=2, invented_facts=["x"]), "JudgeError: bad"])

    assert "ERROR JudgeError: bad" in report
    assert "Agreement on 1 items" in report
