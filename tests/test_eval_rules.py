import json

import pytest

from app.schemas.generate_idea import GenerateIdeaRequest
from evaluation.rules import evaluate, score

REQUEST = GenerateIdeaRequest(
    topic="Ra mắt menu mùa thu", brand_name="Mây Coffee", tone="thân thiện", persona="Sinh viên", content_pillar="Sản phẩm"
)

TITLES = ["Ly cà phê đầu tiên của mùa thu", "Hậu trường pha chế món mới", "Bình chọn đồ uống yêu thích"]


def idea(title: str, text: str = "Nội dung viết bằng tiếng Việt có dấu đầy đủ") -> dict:
    return {"title": title, "approach_angle": text, "hook_sentence": text, "reason": text}


def output(*ideas: dict, **extra) -> str:
    return json.dumps({"ideas": list(ideas), **extra}, ensure_ascii=False)


GOOD = output(*(idea(title) for title in TITLES))


def results_of(raw: str) -> dict[str, bool]:
    return {result.name: result.passed for result in evaluate(REQUEST, raw)}


def test_good_output_passes_every_rule():
    results = evaluate(REQUEST, GOOD)

    assert all(result.passed for result in results)
    assert score(results) == 1.0


def test_not_json_fails_and_skips_the_rest():
    results = evaluate(REQUEST, "Xin lỗi, tôi không thể giúp.")

    assert not any(result.passed for result in results)
    assert all("skipped" in result.detail for result in results[2:])


def test_missing_comma_reports_where_the_json_breaks():
    # Seen with Nova Micro: no comma between the fields of an idea
    raw = '{"ideas": [{"title": "Bí quyết chăm sóc da"\n"approach_angle": "Góc"}]}'

    result = evaluate(REQUEST, raw)[0]

    assert not result.passed
    assert result.detail == "Expecting ',' delimiter at line 2"


def test_code_fence_is_accepted_by_the_service_but_breaks_json_only():
    results = results_of(f"```json\n{GOOD}\n```")

    assert results["valid_json"]
    assert not results["json_only"]
    assert results["schema_valid"]


def test_text_around_the_json_breaks_json_only():
    assert not results_of(f"Đây là kết quả:\n{GOOD}")["json_only"]


def test_json_that_is_not_an_object_fails_both_json_rules():
    results = results_of('["Ý tưởng 1", "Ý tưởng 2", "Ý tưởng 3"]')

    assert not results["valid_json"]
    assert not results["json_only"]


def test_extra_fields_are_reported():
    raw = output(*({**idea(title), "id": 1} for title in TITLES), note="x")

    result = next(r for r in evaluate(REQUEST, raw) if r.name == "no_extra_fields")

    assert not result.passed
    assert result.detail == "id, note"


@pytest.mark.parametrize("raw", [output(idea(TITLES[0]), idea(TITLES[1])), output(idea(""), idea(TITLES[1]), idea(TITLES[2]))])
def test_schema_break_fails_and_skips_content_rules(raw):
    results = results_of(raw)

    assert not results["schema_valid"]
    assert not results["vietnamese_output"]


def test_english_output_fails_vietnamese_rule():
    raw = output(*(idea(title, "This idea is written in English only") for title in TITLES))

    assert not results_of(raw)["vietnamese_output"]


def test_english_brand_name_inside_vietnamese_text_still_passes():
    raw = output(*(idea(title, "Mây Coffee ra mắt Pumpkin Spice Latte cho mùa thu này") for title in TITLES))

    assert results_of(raw)["vietnamese_output"]


@pytest.mark.parametrize("placeholder", ["{{topic}}", "{persona}", "brand_name", "content_pillar"])
def test_copied_placeholder_fails(placeholder):
    raw = output(idea(f"Ưu đãi của {placeholder}"), idea(TITLES[1]), idea(TITLES[2]))

    assert not results_of(raw)["no_placeholder"]


def test_braces_that_are_part_of_the_input_are_not_placeholders():
    request = REQUEST.model_copy(update={"brand_name": "{Mây}"})
    raw = output(idea("Chào mừng đến với {Mây}"), idea(TITLES[1]), idea(TITLES[2]))

    assert all(result.passed for result in evaluate(request, raw))


def test_near_duplicate_titles_fail_distinct_rule():
    raw = output(idea("Menu mùa thu mới"), idea("Menu mùa thu mới!"), idea(TITLES[2]))

    assert not results_of(raw)["distinct_ideas"]
