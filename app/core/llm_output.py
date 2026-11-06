import re


def extract_json(text: str) -> str | None:
    """Tìm khối JSON đầu tiên trong text."""
    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        re.DOTALL,
    )

    if fence_match:
        return fence_match.group(1)

    start = text.find("{")

    if start == -1:
        return None

    depth = 0

    for i, char in enumerate(text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

            if depth == 0:
                return text[start : i + 1]

    return None
