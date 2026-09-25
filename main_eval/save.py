"""Persist experiment results as JSONL + CSV, plus config and metrics files.

Writes, into ``output_dir``:
- ``config.json``          — full run config + template hash + demo IDs/order
- ``template_used.txt``    — the exact template text used
- ``records.jsonl``        — one line per (eval sample, repeat)
- ``results.csv``          — score columns A1..A16 + subtotals + total
- ``metrics.json``         — metrics (only if eval set has human annotations)
- ``agreement.json``       — inter-rater agreement (only if eval set has human annotations)
- ``stability.json``       — repeat-stability summary
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .constants import ITEM_IDS
from .errors import OutputExistsError
from .agreement import compute_agreement
from .metrics import compute_metrics, summarize_repeat_stability

_SUBTOTAL_COLS = ["scene", "episode1", "episode2", "episode3", "total"]

CSV_COLUMNS = [
    "experiment_id", "sample_id", "child_id", "repeat_id", "n_shots", "sample_seed",
    "model", "adapter", "parse_status", "error",
    *ITEM_IDS, *_SUBTOTAL_COLS,
    "model_version", "latency_s", "prompt_tokens", "completion_tokens", "total_tokens",
]


def _ensure_writable(output_dir: Path, force: bool) -> None:
    marker = output_dir / "records.jsonl"
    if output_dir.exists() and marker.exists() and not force:
        raise OutputExistsError(
            f"output dir {output_dir} already contains results; use --force to overwrite"
        )


def save_results(
    records: list[dict],
    *,
    output_dir: str | Path,
    config: dict,
    human_by_sample: dict[str, dict[str, int]] | None = None,
    force: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    _ensure_writable(output_dir, force)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "template_used.txt").write_text(config.get("template", ""), encoding="utf-8")

    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _write_csv(records, output_dir / "results.csv")

    if human_by_sample:
        metrics = compute_metrics(records, human_by_sample)
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        agreement = compute_agreement(records, human_by_sample)
        (output_dir / "agreement.json").write_text(
            json.dumps(agreement, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    stability = summarize_repeat_stability(records)
    (output_dir / "stability.json").write_text(
        json.dumps(stability, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return output_dir


def _write_csv(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = {k: r.get(k) for k in CSV_COLUMNS}
            for iid in ITEM_IDS + _SUBTOTAL_COLS:
                row[iid] = r.get("scores", {}).get(iid, "")
            usage = r.get("usage", {}) or {}
            row["prompt_tokens"] = usage.get("prompt_tokens", "")
            row["completion_tokens"] = usage.get("completion_tokens", "")
            row["total_tokens"] = usage.get("total_tokens", "")
            writer.writerow(row)
