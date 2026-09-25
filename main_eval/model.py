"""Model-call wrapper: retries transport failures, never retries bad answers."""

from __future__ import annotations

from .adapters.base import BaseAdapter, ModelResponse, TransportError

# The distinction that matters for reproducibility:
# - Transport failures (network/API errors) are transient and MAY be retried a
#   bounded number of times.
# - A returned-but-invalid model answer is *never* retried; it is recorded as
#   an invalid output. This is unrelated to research ``repeats`` (sending the
#   same input multiple times and saving each result).


def call_model(
    prompt: str,
    *,
    adapter: BaseAdapter,
    generation_seed: int | None = None,
    settings: dict | None = None,
    max_transport_retries: int = 3,
) -> ModelResponse:
    last_error: str | None = None
    attempts = max(0, max_transport_retries) + 1
    for _ in range(attempts):
        try:
            return adapter.call(prompt, generation_seed=generation_seed, settings=settings)
        except TransportError as exc:
            last_error = str(exc)
    return ModelResponse(raw_text="", usage={}, model_version="", latency_s=0.0, error=last_error)
