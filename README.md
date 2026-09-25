# MAIN_cat_eval

LLM auto-scoring + few-shot experiments for Cantonese MAIN "Cat story" narratives (items A1–A16).

## Installation

- Python 3.9+
- The `openai` SDK is only required for the `openai_compat` adapter:

```bash
pip install openai
```

## Offline check (no API call)

```bash
python -m main_eval run \
  --samples-file data/examples.jsonl \
  --eval-file data/eval.jsonl \
  --prompt-file prompts/main_cat.demo.txt \
  --n-shots 5 --sample-seed 42 --repeats 3 \
  --dry-run --output-dir results/experiment_01
```

## Run against a real model

1. Fill the `{{MAIN_CAT_SCORING_RULES}}` placeholder in `prompts/main_cat.txt` with your official
   MAIN Cat scoring rules.
2. Set your API key from the environment (never hard-coded):

```bash
export OPENAI_API_KEY=sk-...
```

3. Run (omit `--dry-run`):

```bash
python -m main_eval run \
  --samples-file data/examples.jsonl \
  --eval-file data/eval.jsonl \
  --prompt-file prompts/main_cat.txt \
  --n-shots 5 --sample-seed 42 --repeats 3 \
  --model MODEL_NAME --output-dir results/experiment_01
```

## Sweep over multiple conditions

```bash
python -m main_eval sweep \
  --samples-file data/examples.jsonl \
  --eval-file data/eval.jsonl \
  --prompt-file prompts/main_cat.txt \
  --n-shots 0 1 2 5 10 --sample-seeds 11 22 33 --repeats 3 \
  --model MODEL_NAME --output-dir results/experiment_sweep
```

## Recompute agreement from saved results

```bash
python -m main_eval agreement \
  --records-file results/experiment_01/records.jsonl \
  --eval-file data/eval.jsonl
```

## Data format (UTF-8 JSONL)

One child's complete Cat story per line:

```json
{
  "sample_id": "ex_01",
  "child_id": "demo_child_01",
  "story_type": "Cat",
  "transcript": "有一隻貓見到隻雀仔……",
  "annotations": [
    {"item_id": "A1", "score": 2, "evidence": ["quote"], "rationale": "reason"},
    {"item_id": "A2", "score": 1, "evidence": [], "rationale": ""}
  ]
}
```

- `annotations` is required (complete A1–A16) for the demonstration library; optional for the eval
  set (used only for metrics, never sent to the model).
- A1 is scored 0/1/2; A2–A16 are 0/1. The total (0–17) is computed by the code, not the model.

## Key arguments

| Argument | Meaning |
| --- | --- |
| `--samples-file` | human-coded demonstration library (JSONL) |
| `--eval-file` | narratives to score (JSONL) |
| `--prompt-file` | UTF-8 prompt template |
| `--n-shots` | demonstrations per prompt (non-negative integer) |
| `--sample-seed` | seed for demonstration selection/ordering |
| `--repeats` | repeated identical calls per story |
| `--model` | model name (required for `openai_compat`) |
| `--adapter` | `openai_compat` (default) or `mock` |
| `--api-key-env` | env var holding the API key (default `OPENAI_API_KEY`) |
| `--temperature` / `--max-tokens` | inference settings |
| `--generation-seed` | optional model generation seed (independent of sampling seed) |
| `--dry-run` | use the mock adapter (no API call) |
| `--output-dir` | output directory |

## Tests

```bash
python tests/run_all.py
```

## Output

Each run writes, into `--output-dir`: `records.jsonl`, `results.csv`, `config.json`, `metrics.json`,
`agreement.json`, `stability.json`, and `template_used.txt`.

> `data/*.jsonl` is human-constructed demo data, not real child data.
