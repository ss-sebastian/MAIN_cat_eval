import json
import tempfile
from pathlib import Path

from main_eval.adapters import MockAdapter
from main_eval.dataset import load_dataset, validate_dataset
from main_eval.errors import OutputExistsError, PromptError
from main_eval.experiment import run_experiment, run_sweep
from main_eval.prompt import build_prompt, load_prompt_template

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "data" / "examples.jsonl"
EVAL = ROOT / "data" / "eval.jsonl"
TEMPLATE = ROOT / "prompts" / "main_cat.demo.txt"
SKELETON = ROOT / "prompts" / "main_cat.txt"

SETTINGS = {"temperature": 0.0, "max_tokens": 2048}


def _load():
    examples = validate_dataset(load_dataset(EXAMPLES), require_annotations=True, role="examples")
    evals = validate_dataset(load_dataset(EVAL), require_annotations=False, role="eval")
    template = load_prompt_template(TEMPLATE)
    return examples, evals, template


def _run(output_dir, **kw):
    examples, evals, template = _load()
    adapter = MockAdapter()
    base = dict(
        examples=examples, eval_samples=evals, n_shots=2, sample_seed=42, repeats=3,
        model="mock", adapter=adapter, template=template, template_path=str(TEMPLATE),
        settings=SETTINGS, generation_seed=None, max_transport_retries=0,
        output_dir=output_dir, experiment_id="e2e", force=False,
    )
    base.update(kw)
    return run_experiment(**base), evals


def test_run_experiment_with_mock_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        records, evals = _run(d)
        assert len(records) == len(evals) * 3
        assert all(r["parse_status"] == "valid" for r in records)
        assert all(r["scores"]["total"] in range(0, 18) for r in records)
        # no child/sample metadata leakage into the prompt
        for r in records:
            assert r["child_id"] not in r["prompt"]
            assert r["sample_id"] not in r["prompt"]
        for name in ("records.jsonl", "results.csv", "config.json", "metrics.json", "stability.json"):
            assert (Path(d) / name).exists(), name


def test_repeats_identical_input_and_demo_ids():
    with tempfile.TemporaryDirectory() as d:
        records, _ = _run(d, repeats=2)
        by_sample = {}
        for r in records:
            by_sample.setdefault(r["sample_id"], []).append(r)
        for rs in by_sample.values():
            assert len({r["prompt"] for r in rs}) == 1  # identical input per repeat
            assert [r["repeat_id"] for r in rs] == [1, 2]
            assert len({tuple(r["demonstration_ids"]) for r in rs}) == 1  # not re-sampled


def test_nesting_consistent_across_runs():
    with tempfile.TemporaryDirectory() as d:
        r1, _ = _run(Path(d) / "a", n_shots=1, sample_seed=42)
        r2, _ = _run(Path(d) / "b", n_shots=2, sample_seed=42)
        ids1 = r1[0]["demonstration_ids"]
        ids2 = r2[0]["demonstration_ids"]
        assert ids2[:1] == ids1


def test_zero_shot_has_no_demonstrations():
    with tempfile.TemporaryDirectory() as d:
        records, _ = _run(d, n_shots=0, sample_seed=0)
        assert records[0]["demonstration_ids"] == []
        assert "### Example" not in records[0]["prompt"]


def test_sweep_zero_shot_deduplicated():
    examples, evals, template = _load()
    adapter = MockAdapter()
    with tempfile.TemporaryDirectory() as d:
        run_sweep(
            examples, evals, n_shots_list=[0, 1], sample_seeds=[11, 22, 33], repeats=1,
            model="mock", adapter=adapter, template=template, template_path=str(TEMPLATE),
            settings=SETTINGS, generation_seed=None, max_transport_retries=0,
            output_dir=d, force=False,
        )
        summary = json.loads((Path(d) / "sweep_summary.json").read_text(encoding="utf-8"))
        assert len(summary) == 4  # shot_0 once + shot_1 x 3 seeds
        assert len([s for s in summary if s["n_shots"] == 0]) == 1


def test_no_overwrite_without_force():
    with tempfile.TemporaryDirectory() as d:
        _run(d)
        try:
            _run(d)
        except OutputExistsError:
            return
        raise AssertionError("expected OutputExistsError on overwrite without --force")


def test_skeleton_template_errors_until_rules_filled():
    template = load_prompt_template(SKELETON)
    try:
        build_prompt(template, "", "T")
    except PromptError as e:
        assert "MAIN_CAT_SCORING_RULES" in str(e)
        return
    raise AssertionError("expected PromptError because rules placeholder is unfilled")
