# main_cat_story

An LLM auto-scoring + few-shot experiment framework for item-by-item (A1–A16) scoring of
Cantonese MAIN "Cat story" narratives.

The project does **not** train models, does **not** do automatic prompt optimization, and does
**not** use RAG. It uses a single fixed prompt template and varies only the number of
demonstrations (0/1/2/5/10 or any non-negative integer), the sampling seed, and the number of
repeats, to compare scoring quality across few-shot conditions.

---

## Directory layout

```
main_cat_story/
├── main_eval/                 # core package (modular)
│   ├── cli.py                 # main() CLI entry point (run / sweep)
│   ├── __main__.py            # enables `python -m main_eval ...`
│   ├── constants.py           # A1–A16 structure, score ranges, episode grouping
│   ├── schema.py              # Sample / Annotation dataclasses
│   ├── dataset.py             # load_dataset / validate_dataset / validate_child_separation
│   ├── sampling.py            # select_demonstrations (deterministic sampling)
│   ├── prompt.py              # load_prompt_template / format_demonstrations / build_prompt
│   ├── response.py            # parse_and_validate_response
│   ├── scoring.py             # compute_scores (derived subtotals / total)
│   ├── model.py               # call_model (retries transport failures, never retries bad answers)
│   ├── experiment.py          # evaluate_one / run_experiment / run_sweep
│   ├── metrics.py             # compute_metrics / summarize_repeat_stability
│   ├── agreement.py           # inter-rater agreement (kappa / weighted kappa / Krippendorff)
│   ├── save.py                # save_results (JSONL + CSV + config/metrics/agreement/stability)
│   ├── errors.py
│   └── adapters/              # swappable model adapters (mock / openai_compat)
├── prompts/
│   ├── main_cat.txt           # template skeleton (unfilled rules placeholder)
│   └── main_cat.demo.txt      # template with illustrative rules (offline verification only)
├── data/
│   ├── make_demo_data.py      # generates the human-constructed demo data
│   ├── examples.jsonl         # demonstration pool (10 children)
│   └── eval.jsonl             # eval set (2 children, with human gold labels)
├── tests/                     # key tests + standalone runner run_all.py
└── results/                   # run artifacts (JSONL/CSV/config/metrics)
```

---

## Installation

- Python 3.9+ (developed on 3.13).
- The `openai` SDK is only required for the `openai_compat` adapter:

```bash
pip install openai
```

- Run the tests (no third-party dependencies required):

```bash
python tests/run_all.py
# or, if pytest is installed:
python -m pytest tests

---

## Data format (UTF-8 JSONL)

Each record (one line) is one child's complete Cat story narrative:

```json
{
  "sample_id": "ex_01",
  "child_id": "demo_child_01",
  "story_type": "Cat",
  "transcript": "有一隻貓見到隻雀仔……",
  "annotations": [
    {"item_id": "A1", "score": 2, "evidence": ["verbatim quote"], "rationale": "brief reason"},
    {"item_id": "A2", "score": 1, "evidence": [], "rationale": ""}
  ]
}
```

- `sample_id`: stable, unique sample ID.
- `child_id`: child ID. Each child contributes exactly one Cat narrative (duplicates are rejected).
- `story_type`: fixed to `Cat` for this project.
- `transcript`: the complete narrative, **not split into episode segments**.
- `annotations`: per-item annotations, each with `item_id`, `score`, optional `evidence`
  (list of strings) and optional `rationale` (string). **Required and complete (all 16 items) for
  the demonstration pool**; optional for the eval set, but when present they must also be complete
  (used only for metric computation, never sent to the model).

### Score structure

| Group | Items | Scores |
| --- | --- | --- |
| Setting | A1 | 0/1/2 |
| Episode 1 | A2 initiating event IST, A3 goal, A4 attempt, A5 outcome, A6 reaction IST | 0/1 each |
| Episode 2 | A7 initiating event IST, A8 goal, A9 attempt, A10 outcome, A11 reaction IST | 0/1 each |
| Episode 3 | A12 initiating event IST, A13 goal, A14 attempt, A15 outcome, A16 reaction IST | 0/1 each |

Derived fields (computed by Python, never by the model): setting score (A1, 0–2),
episode 1 (A2–A6, 0–5), episode 2 (A7–A11, 0–5), episode 3 (A12–A16, 0–5), total (0–17).

> ⚠️ `data/*.jsonl` is **human-constructed minimal demo data**, not real child speech.
> Do not treat it as real child data.

---

## Running

### Single experiment `run`

```bash
python -m main_eval run \
  --samples-file data/examples.jsonl \
  --eval-file data/eval.jsonl \
  --prompt-file prompts/main_cat.demo.txt \
  --n-shots 5 --sample-seed 42 --repeats 3 \
  --model MODEL_NAME --output-dir results/experiment_01
```

Offline verification (no API call):

```bash
python -m main_eval run \
  --samples-file data/examples.jsonl \
  --eval-file data/eval.jsonl \
  --prompt-file prompts/main_cat.demo.txt \
  --n-shots 5 --sample-seed 42 --repeats 3 \
  --dry-run --output-dir results/experiment_01
```

### Batch `sweep`

```bash
python -m main_eval sweep \
  --samples-file data/examples.jsonl \
  --eval-file data/eval.jsonl \
  --prompt-file prompts/main_cat.demo.txt \
  --n-shots 0 1 2 5 10 --sample-seeds 11 22 33 --repeats 3 \
  --dry-run --output-dir results/experiment_sweep
```

### Argument reference

| Argument | Meaning |
| --- | --- |
| `--samples-file` | human-coded demonstration library (JSONL) |
| `--eval-file` | children's narratives to score (JSONL) |
| `--prompt-file` | UTF-8 text prompt template |
| `--n-shots` | number of full demonstrations per prompt (non-negative integer) |
| `--sample-seed` | random seed controlling demonstration selection/ordering |
| `--repeats` | number of repeated model calls per story |
| `--model` | model name (required for `openai_compat`) |
| `--adapter` | `openai_compat` or `mock` (default `openai_compat`) |
| `--api-key-env` | env var holding the API key (default `OPENAI_API_KEY`) |
| `--base-url` | optional API base URL |
| `--temperature` / `--max-tokens` | inference settings |
| `--generation-seed` | optional **model generation seed** (independent of `--sample-seed`) |
| `--max-transport-retries` | transport-failure retry limit |
| `--dry-run` | use the mock adapter, no API call |
| `--force` | overwrite existing results |
| `--output-dir` | output directory |
| `--experiment-id` | optional experiment-ID override |

### Recompute agreement post-hoc `agreement`

After scores are produced (or without re-running the model), recompute inter-rater agreement from
a saved `records.jsonl` and the eval gold labels:

```bash
python -m main_eval agreement \
  --records-file results/experiment_01/records.jsonl \
  --eval-file data/eval.jsonl \
  --output-file results/experiment_01/agreement.json
```

If `--output-file` is omitted, it writes `<records-dir>/agreement.json`.

### API key

Read from the environment (never written to code, data, or logs):

```bash
export OPENAI_API_KEY=sk-...
# or use a custom env var name:
python -m main_eval run ... --api-key-env MY_KEY_ENV
```

```

---

## Design guarantees (all covered by tests)

1. The demonstration pool is first sorted stably by the unique `sample_id`, then shuffled with a
   local `random.Random(sample_seed)`; the first `k` are taken.
2. Same pool + seed + k ⇒ identical examples in identical order.
3. Nesting: for the same seed, 1-shot is a prefix of 2-shot, and 2-shot is a prefix of 5-shot
   (consistent between `run` and `sweep`).
4. Each condition is sampled exactly once; the selected examples and order are fixed for all eval
   children and all repeats.
5. A child is never used twice in one sample; duplicate `child_id` in the pool is rejected.
6. The demonstration pool and eval set must not share a child (checked before running).
7. `k` exceeding the number of available children is an error (no sampling with replacement).
8. `k=0` uses no demonstrations; in a sweep the zero-shot condition is computed once (not repeated
   per seed).
9. `sample_seed` is used only for demonstration sampling, never as the model generation seed.
10. `repeats=3` sends the **exact same** input three times per story, saving each result; no
    re-sampling and no conversation history.
11. Each inference uses an independent context; no majority voting, no picking the best run.
12. Inference settings are identical across repeats; `--generation-seed` is a separate optional
    parameter (fully reproducible model output is not promised).

Templates are rendered with literal string replacement (not `str.format`), so braces inside JSON
examples are never parsed as placeholders. Missing required placeholders or unreplaced placeholders
(e.g. an unfilled `{{MAIN_CAT_SCORING_RULES}}`) raise an error at run time.

---

## Output artifacts

Each condition directory (`run` writes to `--output-dir`; `sweep` writes to `shot_{k}` /
`shot_{k}_seed_{s}` subdirectories) contains:

- `config.json`: experiment ID, config, template hash, template path, selected demonstration IDs
  and their order, etc.
- `template_used.txt`: the exact template text that was used.
- `records.jsonl`: one record per call (experiment ID, config, template hash, seed, shot count,
  demonstration IDs in order, eval sample_id/child_id, repeat ID, request settings, model version,
  raw response, parse status, A1–A16 scores/evidence/rationale, derived total, token usage, latency,
  error info, and the full prompt).
- `results.csv`: score columns ordered `A1..A16`, `scene`, `episode1`, `episode2`, `episode3`,
  `total`.
- `metrics.json`: when the eval set has human scores — per-item accuracy/confusion, A2–A16
  positive-class precision/recall/F1/support, A1 as a 3-class item, total MAE, mean signed error
  (computed per condition and per repeat, then summarized).
- `agreement.json`: when the eval set has human scores — inter-rater agreement between the model
  predictions and the human gold (see "Inter-rater agreement" below).
- `stability.json`: scoring agreement across repeats of the same input (per-item agreement,
  pairwise disagreement rate, total-score range).
- `sweep_summary.json`: the full list of sweep conditions (subdirectory and record count each).

The terminal does **not** print child transcripts by default — only record / valid / invalid counts.

---

## Inter-rater agreement

`agreement.json` (and the `agreement` subcommand) reports agreement between the **model
predictions** and the **human gold** annotations, treating the model as one rater and the human as
the second. Coefficients (all dependency-free):

- `percent_agreement` — fraction of aligned positions where the two raters agree (0–1).
- `cohens_kappa` — unweighted kappa (chance-corrected); used for the binary items A2–A16.
- `weighted_kappa_linear` / `weighted_kappa_quadratic` — for the ordinal A1 (0–2) and the total
  score (0–17).
- `krippendorff_alpha_nominal` / `krippendorff_alpha_ordinal` — for the total score (and A1),
  generalizing to any number of raters if you later add a second human rater.

Values are in [-1, 1] (1 = perfect, 0 = chance, negative = worse than chance). Agreement is reported
overall, per condition, and per repeat, consistent with `metrics.json`. Only
`parse_status == "valid"` records on labeled samples contribute.

---

## Metric conventions

- Invalid outputs (missing items, duplicate items, out-of-range scores, non-integer scores) are
  **never defaulted to zero and never silently corrected**; they are counted separately under
  `invalid_outputs`, and the valid-sample denominator is reported explicitly in `metrics.json`.
- Per-item accuracy and confusion matrices are computed only over `parse_status == "valid"` records
  that have human annotations.
- Repeat stability compares only scores; identical scores with different rationale wording are not
  counted as disagreement. With a single repeat, stability is marked not estimable.
- Repeated predictions are not treated as new independent children, and no unplanned significance
  testing is performed.
- Transport-failure retries (`--max-transport-retries`) are strictly separate from research
  `repeats`; invalid answers are never retried indefinitely.

---

## Steps before using real data

1. Replace the `{{MAIN_CAT_SCORING_RULES}}` placeholder in `prompts/main_cat.txt` with the
   official MAIN Cat manual's per-item boundary-scoring rules; running with it unfilled raises an
   error.
2. Prepare your own demonstration library and eval set (matching the JSONL schema above).
3. Set `OPENAI_API_KEY` (or a custom env var).
4. Verify offline with `--dry-run` before making real API calls.

> The bundled `prompts/main_cat.demo.txt` only states the item structure and score ranges (taken
> from the project specification). It contains **no** per-item boundary-scoring rules and is
> intended solely for offline mock verification; replace it with the official rules before any real
> experiment.

