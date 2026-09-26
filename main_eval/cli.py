"""Command-line interface: ``run`` and ``sweep`` subcommands."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .adapters import build_adapter
from .agreement import compute_agreement
from .dataset import annotations_map_by_sample, load_dataset, validate_dataset
from .errors import MainEvalError
from .experiment import run_experiment, run_sweep
from .prompt import load_prompt_template
from .registry import load_registry, models_by_tier, resolve_model


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return n


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main_eval",
        description="LLM auto-scoring + few-shot experiments for Cantonese MAIN Cat stories.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--samples-file", required=True, help="human-coded demonstration library (JSONL)")
        p.add_argument("--eval-file", required=True, help="children's narratives to score (JSONL)")
        p.add_argument("--prompt-file", required=True, help="UTF-8 prompt template")
        p.add_argument("--repeats", type=_positive_int, required=True, help="identical model calls per story")
        model_group = p.add_mutually_exclusive_group()
        model_group.add_argument("--model", default=None,
                                 help="model name or registry alias (required unless mock/dry-run)")
        model_group.add_argument("--models", nargs="+", default=None,
                                 help="run a series: one or more model names / registry aliases")
        model_group.add_argument("--model-tier", default=None,
                                 help="run a series: all registry models with this tier "
                                      "(frontier|reasoning|zh|cheap)")
        p.add_argument("--models-file", default="models/registry.json",
                       help="path to the model registry JSON")
        p.add_argument("--adapter", default="openai_compat", choices=["openai_compat", "mock"])
        p.add_argument("--api-key-env", default="OPENAI_API_KEY", help="fallback env var holding the API key")
        p.add_argument("--base-url", default=None, help="optional API base URL (overrides registry)")
        p.add_argument("--temperature", type=float, default=0.0)
        p.add_argument("--max-tokens", type=int, default=2048)
        p.add_argument("--generation-seed", type=int, default=None,
                       help="optional model generation seed (independent of sample seed)")
        p.add_argument("--max-transport-retries", type=_positive_int, default=3)
        p.add_argument("--dry-run", action="store_true", help="use mock adapter, no API call")
        p.add_argument("--force", action="store_true", help="overwrite existing results")
        p.add_argument("--output-dir", required=True, help="output directory")
        p.add_argument("--experiment-id", default=None, help="optional override for experiment id")

    run_p = sub.add_parser("run", help="run a single (n-shots, sample-seed) condition")
    run_p.add_argument("--n-shots", type=_positive_int, required=True)
    run_p.add_argument("--sample-seed", type=int, required=True)
    add_common(run_p)

    sweep_p = sub.add_parser("sweep", help="run multiple shot counts x sample seeds")
    sweep_p.add_argument("--n-shots", type=_positive_int, nargs="+", required=True)
    sweep_p.add_argument("--sample-seeds", type=int, nargs="+", required=True)
    add_common(sweep_p)

    agree_p = sub.add_parser(
        "agreement",
        help="recompute inter-rater agreement from saved records + eval gold",
    )
    agree_p.add_argument("--records-file", required=True, help="path to a records.jsonl")
    agree_p.add_argument("--eval-file", required=True, help="eval JSONL with human gold annotations")
    agree_p.add_argument("--output-file", default=None,
                         help="output path (default: <records-dir>/agreement.json)")

    return parser

def _settings(args) -> dict:
    return {"temperature": args.temperature, "max_tokens": args.max_tokens}


def _build_model_config(args, registry, key):
    """Resolve one model (alias or raw name) into an adapter + metadata."""
    adapter_name = "mock" if bool(args.dry_run) else args.adapter

    if key is None:
        resolved = {"key": "mock", "model": "mock", "provider": None,
                    "base_url": None, "api_key_env": None, "params": {}, "tier": None}
    else:
        resolved = resolve_model(registry, key)

    model = resolved["model"]
    base_url = resolved["base_url"] or args.base_url
    api_key_env = resolved["api_key_env"] or args.api_key_env
    params = resolved["params"]

    if adapter_name == "openai_compat":
        if not model:
            raise SystemExit("--model is required when using the openai_compat adapter")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise SystemExit(f"API key not found: set environment variable {api_key_env}")
        adapter = build_adapter("openai_compat", model=model, api_key=api_key,
                                base_url=base_url, params=params)
    else:
        model = model or "mock"
        adapter = build_adapter("mock", model=model, api_key="")

    return {
        "key": resolved["key"],
        "model": model,
        "adapter": adapter,
        "meta": {"key": resolved["key"], "provider": resolved["provider"], "base_url": base_url},
    }


def _resolve_model_configs(args) -> list[dict]:
    """Return the list of model configs to run, in order."""
    registry = load_registry(args.models_file)
    if args.models:
        keys = list(args.models)
    elif args.model_tier:
        keys = models_by_tier(registry, args.model_tier)
        if not keys:
            raise SystemExit(f"no models with tier {args.model_tier!r} in {args.models_file}")
    elif args.model:
        keys = [args.model]
    else:
        keys = [None]  # default: mock (only valid with --dry-run / --adapter mock)

    return [_build_model_config(args, registry, k) for k in keys]


def _load(args):
    examples = validate_dataset(load_dataset(args.samples_file), require_annotations=True, role="examples")
    evals = validate_dataset(load_dataset(args.eval_file), require_annotations=False, role="eval")
    template = load_prompt_template(args.prompt_file)
    return examples, evals, template


def _summary(records: list[dict]) -> str:
    n = len(records)
    valid = sum(1 for r in records if r["parse_status"] == "valid")
    return f"records={n} valid={valid} invalid={n - valid}"


def cmd_run(args) -> int:
    examples, evals, template = _load(args)
    settings = _settings(args)
    configs = _resolve_model_configs(args)

    for cfg in configs:
        subdir = args.output_dir if len(configs) == 1 else str(Path(args.output_dir) / cfg["key"])
        experiment_id = args.experiment_id or (
            f"main_cat_{cfg['key']}_shots{args.n_shots}_seed{args.sample_seed}"
        )
        records = run_experiment(
            examples,
            evals,
            n_shots=args.n_shots,
            sample_seed=args.sample_seed,
            repeats=args.repeats,
            model=cfg["model"],
            adapter=cfg["adapter"],
            template=template,
            template_path=args.prompt_file,
            settings=settings,
            generation_seed=args.generation_seed,
            max_transport_retries=args.max_transport_retries,
            output_dir=subdir,
            experiment_id=experiment_id,
            force=args.force,
            model_meta=cfg["meta"],
        )
        print(f"[run] model={cfg['key']} {_summary(records)}")
        print(f"[run] wrote results to {subdir}")
    return 0


def cmd_sweep(args) -> int:
    examples, evals, template = _load(args)
    settings = _settings(args)
    configs = _resolve_model_configs(args)

    for cfg in configs:
        subdir = args.output_dir if len(configs) == 1 else str(Path(args.output_dir) / cfg["key"])
        records = run_sweep(
            examples,
            evals,
            n_shots_list=args.n_shots,
            sample_seeds=args.sample_seeds,
            repeats=args.repeats,
            model=cfg["model"],
            adapter=cfg["adapter"],
            template=template,
            template_path=args.prompt_file,
            settings=settings,
            generation_seed=args.generation_seed,
            max_transport_retries=args.max_transport_retries,
            output_dir=subdir,
            force=args.force,
            model_meta=cfg["meta"],
        )
        n_conditions = sum(len(args.sample_seeds) if k else 1 for k in args.n_shots)
        print(f"[sweep] model={cfg['key']} conditions={n_conditions} {_summary(records)}")
        print(f"[sweep] wrote results to {subdir}")
    return 0


def cmd_agreement(args) -> int:
    records_path = Path(args.records_file)
    with records_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    evals = validate_dataset(load_dataset(args.eval_file), require_annotations=False, role="eval")
    human = annotations_map_by_sample(evals)
    if not human:
        print("warning: eval file has no human annotations; agreement not computable", file=sys.stderr)

    result = compute_agreement(records, human)

    out = Path(args.output_file) if args.output_file else records_path.parent / "agreement.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = result.get("overall")
    if overall:
        print(
            f"[agreement] n_valid={result['n_valid']} "
            f"macro_cohens_kappa={overall.get('macro_cohens_kappa')} "
            f"total_weighted_kappa_quadratic={overall['total']['weighted_kappa_quadratic']}"
        )
    print(f"[agreement] wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return cmd_run(args)
        if args.command == "sweep":
            return cmd_sweep(args)
        if args.command == "agreement":
            return cmd_agreement(args)
    except MainEvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
