"""Dataset loading and validation (pure functions, no model/API dependency)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from .constants import ITEM_IDS, ITEM_SCORES, STORY_TYPE
from .errors import ChildSeparationError, DatasetError
from .schema import Annotation, Sample


def load_dataset(path: str | Path) -> list[dict]:
    """Read a UTF-8 JSONL file and return raw records (list of dicts).

    Raises DatasetError on a malformed JSON line. Semantic validation is left
    to :func:`validate_dataset`.
    """
    path = Path(path)
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:  # noqa: PERF203
                raise DatasetError(f"{path}: line {lineno} is not valid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise DatasetError(f"{path}: line {lineno} must be a JSON object, got {type(obj).__name__}")
            records.append(obj)
    return records


def _validate_annotations(rec: dict, sample_id: str) -> list[Annotation]:
    anns = rec.get("annotations")
    if anns is None:
        return []
    if not isinstance(anns, list):
        raise DatasetError(f"{sample_id}: 'annotations' must be a list")
    out: list[Annotation] = []
    seen: set[str] = set()
    for a in anns:
        if not isinstance(a, dict):
            raise DatasetError(f"{sample_id}: each annotation must be an object")
        item_id = a.get("item_id")
        if item_id not in ITEM_IDS:
            raise DatasetError(f"{sample_id}: unknown item_id {item_id!r}")
        if item_id in seen:
            raise DatasetError(f"{sample_id}: duplicate item_id {item_id!r} in annotations")
        seen.add(item_id)
        score = a.get("score")
        if isinstance(score, bool) or not isinstance(score, int):
            raise DatasetError(f"{sample_id}/{item_id}: score must be an integer, got {score!r}")
        if score not in ITEM_SCORES[item_id]:
            raise DatasetError(f"{sample_id}/{item_id}: score {score} out of range {ITEM_SCORES[item_id]}")
        evidence = a.get("evidence", [])
        if not isinstance(evidence, list):
            raise DatasetError(f"{sample_id}/{item_id}: 'evidence' must be a list")
        if any(not isinstance(e, str) for e in evidence):
            raise DatasetError(f"{sample_id}/{item_id}: every evidence entry must be a string")
        rationale = a.get("rationale", "")
        if rationale is None:
            rationale = ""
        if not isinstance(rationale, str):
            raise DatasetError(f"{sample_id}/{item_id}: 'rationale' must be a string")
        out.append(Annotation(item_id=item_id, score=score, evidence=list(evidence), rationale=rationale))
    return out


def validate_dataset(
    records: Sequence[dict],
    *,
    require_annotations: bool,
    role: str = "dataset",
) -> list[Sample]:
    """Validate raw records and return typed Sample objects.

    Checks required fields, story type, unique sample_id, unique child_id
    (each child contributes exactly one Cat narrative), and annotation
    shape/range. When ``require_annotations`` is True (demonstration pool),
    annotations must be present and complete (all 16 items). When annotations
    are present but not required (eval set), they must still be complete so
    metrics can be computed.
    """
    samples: list[Sample] = []
    seen_sample_ids: set[str] = set()
    seen_child_ids: set[str] = set()

    for rec in records:
        missing = [f for f in ("sample_id", "child_id", "story_type", "transcript") if f not in rec]
        if missing:
            raise DatasetError(f"{role}: record missing required field(s): {', '.join(missing)}")

        sample_id = rec["sample_id"]
        child_id = rec["child_id"]
        story_type = rec["story_type"]
        transcript = rec["transcript"]

        if not isinstance(sample_id, str) or not sample_id.strip():
            raise DatasetError(f"{role}: 'sample_id' must be a non-empty string")
        if not isinstance(child_id, str) or not child_id.strip():
            raise DatasetError(f"{role}/{sample_id}: 'child_id' must be a non-empty string")
        if story_type != STORY_TYPE:
            raise DatasetError(f"{role}/{sample_id}: story_type must be {STORY_TYPE!r}, got {story_type!r}")
        if not isinstance(transcript, str) or not transcript.strip():
            raise DatasetError(f"{role}/{sample_id}: 'transcript' must be a non-empty string")

        if sample_id in seen_sample_ids:
            raise DatasetError(f"{role}: duplicate sample_id {sample_id!r}")
        if child_id in seen_child_ids:
            raise DatasetError(f"{role}: duplicate child_id {child_id!r} (each child may have only one Cat narrative)")
        seen_sample_ids.add(sample_id)
        seen_child_ids.add(child_id)

        annotations = _validate_annotations(rec, sample_id)

        if annotations:
            present = {a.item_id for a in annotations}
            if present != set(ITEM_IDS):
                missing_items = sorted(set(ITEM_IDS) - present)
                raise DatasetError(
                    f"{role}/{sample_id}: annotations incomplete, missing items {missing_items}"
                )
        elif require_annotations:
            raise DatasetError(f"{role}/{sample_id}: demonstration records require complete annotations")

        samples.append(
            Sample(
                sample_id=sample_id,
                child_id=child_id,
                story_type=story_type,
                transcript=transcript,
                annotations=annotations,
            )
        )

    return samples


def validate_child_separation(examples: Sequence[Sample], evals: Sequence[Sample]) -> None:
    """Ensure the demonstration pool and eval set share no child.

    Raises ChildSeparationError listing the offending child IDs.
    """
    example_children = {s.child_id for s in examples}
    eval_children = {s.child_id for s in evals}
    overlap = example_children & eval_children
    if overlap:
        raise ChildSeparationError(
            "demonstration pool and eval set share child_id(s): " + ", ".join(sorted(overlap))
        )


def annotations_map_by_sample(samples: Iterable[Sample]) -> dict[str, dict[str, int]]:
    """Return {sample_id: {item_id: score}} for samples that have annotations."""
    return {s.sample_id: s.annotation_map() for s in samples if s.annotations}

