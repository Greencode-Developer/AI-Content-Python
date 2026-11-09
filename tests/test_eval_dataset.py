from collections import Counter

from evaluation.dataset import load_dataset

CASES = load_dataset()


def test_dataset_loads_into_the_request_schema():
    # load_dataset validates every line against EvalCase / GenerateIdeaRequest
    assert len(CASES) >= 20


def test_case_ids_are_unique():
    duplicates = [case_id for case_id, count in Counter(case.id for case in CASES).items() if count > 1]

    assert duplicates == []


def test_every_category_is_covered():
    assert {case.category for case in CASES} == {"normal", "edge", "hard"}
