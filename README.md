# AI Content Python Service

```bash
uv sync
cp .env.example .env   # then fill in the keys
uv run fastapi dev app/main.py
```

Tests:

```bash
uv run pytest
```

Evaluation (calls the real model, needs the Langfuse and provider keys in `.env`):

```bash
uv run python -m evaluation.run --provider gemini --limit 5
uv run python -m evaluation.run --provider bedrock --category hard
```

The dataset is `evaluation/datasets/generate_ideas.jsonl`. Each case is called once
without retry and checked by the rules in `evaluation/rules.py`. The summary is printed
and the full output is saved to `evaluation/results/` (not committed).
