"""Deterministic mock adapter for --dry-run / offline testing.

Produces a *valid* A1-A16 JSON deterministically from the prompt text, so the
full pipeline (load -> validate -> sample -> prompt -> parse -> score -> save)
can run without any API call. Repeats of the same prompt return identical
outputs, which makes repeat-stability tests meaningful.
"""

from __future__ import annotations

import hashlib
import json

from ..constants import ITEM_IDS
from .base import BaseAdapter, ModelResponse


class MockAdapter(BaseAdapter):
    name = "mock"

    def call(
        self,
        prompt: str,
        *,
        generation_seed: int | None = None,
        settings: dict | None = None,
    ) -> ModelResponse:
        del generation_seed, settings  # deterministic; seed/settings ignored by design
        h = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
        items = []
        for idx, item_id in enumerate(ITEM_IDS):
            r = h + idx
            score = r % 3 if item_id == "A1" else r % 2
            items.append(
                {
                    "item_id": item_id,
                    "score": score,
                    "evidence": [],
                    "rationale": "mock rationale (deterministic, no API call)",
                }
            )
        raw = json.dumps({"items": items}, ensure_ascii=False)
        # Rough token estimates so usage fields are populated for CSV/records.
        usage = {
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": len(raw) // 4,
            "total_tokens": (len(prompt) + len(raw)) // 4,
        }
        return ModelResponse(
            raw_text=raw,
            usage=usage,
            model_version="mock-0.1.0",
            latency_s=0.0,
        )
