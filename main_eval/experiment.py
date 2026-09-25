"""Experiment orchestration: evaluate_one, run_experiment, run_sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .adapters.base import BaseAdapter
from .dataset import annotations_map_by_sample, validate_child_separation
from .model import call_model
from .prompt import build_prompt, format_demonstrations, template_hash
from .response import ParsedResponse, parse_and_validate_response
from .sampling import select_demonstrations
from .save import save_results
from .schema import Sample
from .scoring import compute_scores


def evaluate_one(
    eval_sample: Sample,
    demonstrations_text: str,
    template: str,
    *,
    ctx: dict,
    adapter: BaseAdapter,
    repeats: int,
    generation_seed: int | None,
    settings: dict,
    max_transport_retries: int,
) -> list[dict]:
    """Run ``repeats`` independent calls for one eval sample.

    Each repeat builds the *same* prompt and sends it as a fresh, independent
    request (no conversation history, no re-sampling, no answer carried over).
    Eval human annotations are never touched here (metrics use them afterwards).
    """
    prompt = build_prompt(template, demonstrations_text, eval_sample.transcript)
    records: list[dict] = []
    for r in range(1, repeats + 1):
        resp = call_model(
            prompt,
            adapter=adapter,
            generation_seed=generation_seed,
            settings=settings,
            max_transport_retries=max_transport_retries,
        )
        if resp.error is None:
            parsed = parse_and_validate_response(resp.raw_text)
        else:
            parsed = ParsedResponse(valid=False, error=f"transport error: {resp.error}")
        scores = compute_scores(parsed.scores) if parsed.valid else {}
        records.append(
            {
                "experiment_id": ctx["experiment_id"],
                "n_shots": ctx["n_shots"],
                "sample_seed": ctx["sample_seed"],
                "sample_id": eval_sample.sample_id,
                "child_id": eval_sample.child_id,
                "repeat_id": r,
                "model": ctx["model"],
                "adapter": ctx["adapter"],
                "template_hash": ctx["template_hash"],
                "template_path": ctx["template_path"],
                "demonstration_ids": ctx["demonstration_ids"],
                "settings": ctx["settings"],
                "generation_seed": generation_seed,
                "prompt": prompt,
                "model_version": resp.model_version,
                "raw_response": resp.raw_text,
                "parse_status": "valid" if parsed.valid else "invalid",
                "parse_error": parsed.error,
                "items": parsed.items,
                "scores": scores,
                "usage": resp.usage,
                "latency_s": resp.latency_s,
                "error": resp.error,
            }
        )
    return records


def run_experiment(
    examples: Sequence[Sample],
    eval_samples: Sequence[Sample],
    *,
    n_shots: int,
    sample_seed: int,
    repeats: int,
    model: str,
    adapter: BaseAdapter,
    template: str,
    template_path: str,
    settings: dict,
    generation_seed: int | None,
    max_transport_retries: int,
    output_dir: str | Path,
    experiment_id: str,
    force: bool = False,
) -> list[dict]:
    """Run one (n_shots, sample_seed) condition and save results."""
    validate_child_separation(examples, eval_samples)
    demos = select_demonstrations(examples, sample_seed, n_shots)
    demos_text = format_demonstrations(demos)
    demo_ids = [d.sample_id for d in demos]

    th = template_hash(template)
    ctx = {
        "experiment_id": experiment_id,
        "n_shots": n_shots,
        "sample_seed": sample_seed,
        "model": model,
        "adapter": adapter.name,
        "template_hash": th,
        "template_path": str(template_path),
        "demonstration_ids": demo_ids,
        "settings": settings,
    }

    records: list[dict] = []
    for ev in eval_samples:
        records.extend(
            evaluate_one(
                ev,
                demos_text,
                template,
                ctx=ctx,
                adapter=adapter,
                repeats=repeats,
                generation_seed=generation_seed,
                settings=settings,
                max_transport_retries=max_transport_retries,
            )
        )

    config = {
        "experiment_id": experiment_id,
        "n_shots": n_shots,
        "sample_seed": sample_seed,
        "repeats": repeats,
        "model": model,
        "adapter": adapter.name,
        "settings": settings,
        "generation_seed": generation_seed,
        "max_transport_retries": max_transport_retries,
        "template_path": str(template_path),
        "template_hash": th,
        "template": template,
        "demonstration_ids": demo_ids,
        "n_examples": len(examples),
        "n_eval_samples": len(eval_samples),
    }

    human = annotations_map_by_sample(eval_samples)
    save_results(records, output_dir=output_dir, config=config, human_by_sample=human, force=force)
    return records



def _iter_conditions(n_shots_list: Sequence[int], sample_seeds: Sequence[int]):
    """Yield (n_shots, sample_seed) conditions, deduplicating zero-shot.

    k=0 uses no demonstrations, so it is computed once regardless of how many
    sample seeds are given (requirement: don't recompute the same zero-shot
    condition for multiple sample seeds).
    """
    for k in n_shots_list:
        if k == 0:
            yield (0, None)
        else:
            for seed in sample_seeds:
                yield (k, seed)


def run_sweep(
    examples: Sequence[Sample],
    eval_samples: Sequence[Sample],
    *,
    n_shots_list: Sequence[int],
    sample_seeds: Sequence[int],
    repeats: int,
    model: str,
    adapter: BaseAdapter,
    template: str,
    template_path: str,
    settings: dict,
    generation_seed: int | None,
    max_transport_retries: int,
    output_dir: str | Path,
    force: bool = False,
) -> list[dict]:
    """Run every (n_shots, sample_seed) condition under a shared output dir."""
    output_dir = Path(output_dir)
    all_records: list[dict] = []
    summary = []
    for k, seed in _iter_conditions(n_shots_list, sample_seeds):
        if seed is None:
            subdir = output_dir / f"shot_{k}"
            label = f"shots={k}"
        else:
            subdir = output_dir / f"shot_{k}_seed_{seed}"
            label = f"shots={k},seed={seed}"
        eid = f"main_cat_{label.replace(',', '_').replace('=', '')}"
        records = run_experiment(
            examples,
            eval_samples,
            n_shots=k,
            sample_seed=0 if seed is None else seed,
            repeats=repeats,
            model=model,
            adapter=adapter,
            template=template,
            template_path=template_path,
            settings=settings,
            generation_seed=generation_seed,
            max_transport_retries=max_transport_retries,
            output_dir=subdir,
            experiment_id=eid,
            force=force,
        )
        all_records.extend(records)
        summary.append(
            {
                "condition": label,
                "n_shots": k,
                "sample_seed": seed,
                "output_dir": str(subdir),
                "n_records": len(records),
            }
        )

    (output_dir / "sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return all_records
