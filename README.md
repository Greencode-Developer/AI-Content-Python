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
uv run python -m evaluation.run --provider bedrock --limit 5
uv run python -m evaluation.run --provider bedrock --category hard
```

The dataset is `evaluation/datasets/generate_ideas.jsonl`. Each case is called once
without retry and checked by the rules in `evaluation/rules.py`. The summary is printed
and the full output is saved to `evaluation/results/` (not committed).

LLM judge (Claude on Bedrock, `sonnet-4.6` by default) scores what the rules cannot check:
natural Vietnamese, invented facts, safety issues, relevance, specificity and persona fit.

```bash
uv run python -m evaluation.run --judge                      # generate, then judge
uv run python -m evaluation.run --from-results evaluation/results/<file>.json --judge --judge-model haiku-4.5
```

`--from-results` re-scores the outputs of a saved run, so judges can be compared on the same output.

To test a new prompt version before it goes to production, give it a Langfuse label such as
`candidate` and run `uv run python -m evaluation.run --prompt-label candidate --judge`.
The label and version are saved with the results.

Before trusting a judge, fill the `human` labels in `evaluation/datasets/judge_calibration.json`
and compare:

```bash
uv run python -m evaluation.calibrate --judge-model sonnet-4.6
```
