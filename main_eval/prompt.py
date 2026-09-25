"""Prompt-template loading and rendering.

The template is plain UTF-8 text with the two dynamic placeholders
``{{demonstrations}}`` and ``{{transcript}}``. Rendering uses literal string
replacement (NOT ``str.format``), so single braces appearing inside JSON
examples in the template or in demonstration data are never interpreted as
placeholders.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

from .constants import TEMPLATE_PLACEHOLDERS
from .errors import PromptError
from .schema import Sample

# Matches any remaining {{ ... }} token after the two known placeholders are
# replaced. Used to detect unfilled placeholders (e.g. a rules placeholder).
_PLACEHOLDER_RE = re.compile(r"\{\{.*?\}\}", re.DOTALL)


def load_prompt_template(path: str | Path) -> str:
    """Read a UTF-8 prompt template and validate required placeholders."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    for ph in TEMPLATE_PLACEHOLDERS:
        if ph not in text:
            raise PromptError(f"template {path} is missing required placeholder {ph!r}")
    return text


def template_hash(text: str) -> str:
    """SHA-256 hex digest of the template content (for run records)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _annotations_to_json(sample: Sample) -> str:
    items = [a.to_dict() for a in sample.annotations]
    return json.dumps({"items": items}, ensure_ascii=False, indent=2)


def format_demonstrations(examples: Sequence[Sample]) -> str:
    """Render demonstrations using only the whitelisted fields (transcript +
    annotations). sample_id/child_id/story_type are intentionally excluded.
    """
    if not examples:
        return ""
    blocks = []
    for i, ex in enumerate(examples, 1):
        blocks.append(
            "### Example " + str(i) + "\n"
            "Transcript:\n"
            + ex.transcript.strip()
            + "\n\nHuman-verified scores:\n```json\n"
            + _annotations_to_json(ex)
            + "\n```"
        )
    return "\n\n".join(blocks)


def find_unreplaced_placeholders(text: str) -> list[str]:
    """Return any remaining ``{{ ... }}`` tokens in a rendered prompt."""
    return _PLACEHOLDER_RE.findall(text)


def build_prompt(template: str, demonstrations_text: str, transcript: str) -> str:
    """Render the final prompt.

    Replaces the two dynamic placeholders and errors if any ``{{ ... }}``
    token remains (e.g. an unfilled rules placeholder such as
    ``{{MAIN_CAT_SCORING_RULES}}``).
    """
    prompt = template.replace("{{demonstrations}}", demonstrations_text)
    prompt = prompt.replace("{{transcript}}", transcript)
    leftover = find_unreplaced_placeholders(prompt)
    if leftover:
        raise PromptError(
            "unreplaced template placeholder(s) detected: "
            + ", ".join(sorted(set(leftover)))
            + " (fill in the required rules/rubric before running)"
        )
    return prompt
