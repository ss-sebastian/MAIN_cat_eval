"""OpenAI-compatible Chat Completions adapter.

The ``openai`` SDK is imported lazily inside ``__init__`` so that importing this
module (and the rest of the package) never fails when the SDK is absent.

API credentials are read from the environment at construction time; they are
never written to code, data files, or logs.
"""

from __future__ import annotations

import time

from .base import BaseAdapter, ModelResponse, TransportError


class OpenAICompatibleAdapter(BaseAdapter):
    name = "openai_compat"

    def __init__(self, model: str, api_key: str, base_url: str | None = None, params: dict | None = None):
        try:
            from openai import OpenAI  # lazy import
        except ImportError as exc:  # pragma: no cover - exercised only when SDK missing
            raise TransportError(
                "the 'openai' package is required for the openai_compat adapter; "
                "pip install openai, or use --adapter mock / --dry-run"
            ) from exc
        self.model = model
        self.params = params or {}
        self._client = OpenAI(api_key=api_key, base_url=base_url or None)

    def call(
        self,
        prompt: str,
        *,
        generation_seed: int | None = None,
        settings: dict | None = None,
    ) -> ModelResponse:
        settings = settings or {}
        p = self.params
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        # Apply a per-model parameter policy so that providers which reject or
        # rename certain parameters (e.g. reasoning models) still work.
        if "temperature" in settings and not p.get("drop_temperature"):
            kwargs["temperature"] = settings["temperature"]
        if "max_tokens" in settings:
            if p.get("use_max_completion_tokens"):
                kwargs["max_completion_tokens"] = settings["max_tokens"]
            elif not p.get("drop_max_tokens"):
                kwargs["max_tokens"] = settings["max_tokens"]
        if generation_seed is not None and not p.get("drop_seed"):
            kwargs["seed"] = generation_seed
        if p.get("extra_body"):
            kwargs["extra_body"] = p["extra_body"]

        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # network / API errors -> retryable transport failure
            raise TransportError(f"chat.completions.create failed: {exc}") from exc
        latency = time.perf_counter() - start

        choice = resp.choices[0] if resp.choices else None
        content = (choice.message.content if choice and choice.message else "") or ""
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
        return ModelResponse(
            raw_text=content,
            usage=usage,
            model_version=getattr(resp, "model", ""),
            latency_s=latency,
        )
