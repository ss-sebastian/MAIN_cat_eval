"""Parse and validate a model's JSON response against the A1-A16 schema.

Strictness rules (no silent fixing / no defaulting to zero):
- Missing items, duplicate items, out-of-range or non-integer scores all mark
  the output invalid.
- Zero-score items must still be present.
- ``evidence`` must be a list of strings; missing -> empty list. An empty list
  is valid (no fabricated citations).
- ``rationale`` must be a string; missing -> empty string.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .constants import ITEM_IDS, ITEM_SCORES


@dataclass
class ParsedResponse:
    valid: bool
    items: list[dict] = field(default_factory=list)  # A1..A16 order, only when valid
    scores: dict[str, int] = field(default_factory=dict)  # item_id -> score (valid only)
    error: str | None = None


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced top-level JSON object substring, if any."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json(text: str) -> Any:
    """Attempt to parse the raw text as JSON (tolerating markdown fences)."""
    text = text.strip()
    candidates = [text]
    # Strip a single markdown code fence if present.
    stripped = text
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        stripped = stripped.rsplit("```", 1)[0]
        candidates.append(stripped.strip())
    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
    obj = _extract_json_object(text)
    if obj is not None:
        try:
            return json.loads(obj)
        except (json.JSONDecodeError, ValueError):
            pass
    raise ValueError("no JSON object found in response")


def parse_and_validate_response(raw_text: str) -> ParsedResponse:
    if raw_text is None:
        raw_text = ""
    try:
        data = _extract_json(raw_text)
    except ValueError as exc:
        return ParsedResponse(valid=False, error=f"could not parse JSON: {exc}")

    if not isinstance(data, dict):
        return ParsedResponse(valid=False, error="response JSON is not an object")
    items = data.get("items")
    if not isinstance(items, list):
        return ParsedResponse(valid=False, error="response JSON missing 'items' list")

    by_id: dict[str, dict] = {}
    errors: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            errors.append("an item is not an object")
            continue
        item_id = it.get("item_id")
        if item_id not in ITEM_IDS:
            errors.append(f"unknown item_id {item_id!r}")
            continue
        if item_id in by_id:
            errors.append(f"duplicate item_id {item_id!r}")
            continue
        score = it.get("score")
        if isinstance(score, bool) or not isinstance(score, int):
            errors.append(f"{item_id}: score is not an integer ({score!r})")
            continue
        if score not in ITEM_SCORES[item_id]:
            errors.append(f"{item_id}: score {score} out of range {ITEM_SCORES[item_id]}")
            continue
        evidence = it.get("evidence", [])
        if evidence is None:
            evidence = []
        if not isinstance(evidence, list) or any(not isinstance(e, str) for e in evidence):
            errors.append(f"{item_id}: 'evidence' must be a list of strings")
            continue
        rationale = it.get("rationale", "")
        if rationale is None:
            rationale = ""
        if not isinstance(rationale, str):
            errors.append(f"{item_id}: 'rationale' must be a string")
            continue
        by_id[item_id] = {
            "item_id": item_id,
            "score": score,
            "evidence": list(evidence),
            "rationale": rationale,
        }

    if errors:
        return ParsedResponse(valid=False, error="; ".join(errors))

    # Completeness + canonical ordering.
    missing = [iid for iid in ITEM_IDS if iid not in by_id]
    if missing:
        return ParsedResponse(valid=False, error="missing item_id(s): " + ", ".join(missing))

    ordered = [by_id[iid] for iid in ITEM_IDS]
    scores = {iid: by_id[iid]["score"] for iid in ITEM_IDS}
    return ParsedResponse(valid=True, items=ordered, scores=scores, error=None)
