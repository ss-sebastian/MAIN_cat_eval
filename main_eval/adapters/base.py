"""Model adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ModelResponse:
    """Structured result of a single model call."""

    raw_text: str = ""
    usage: dict = field(default_factory=dict)
    model_version: str = ""
    latency_s: float = 0.0
    error: str | None = None


class TransportError(Exception):
    """A transient API/network failure (safe to retry)."""


class BaseAdapter(ABC):
    """Abstract model-service adapter.

    Business code depends only on this interface, never on a concrete SDK.
    """

    name: str = "base"

    @abstractmethod
    def call(
        self,
        prompt: str,
        *,
        generation_seed: int | None = None,
        settings: dict | None = None,
    ) -> ModelResponse:
        """Send ``prompt`` and return a ModelResponse.

        ``generation_seed`` is optional and, when supported, is passed to the
        service as its own generation seed. It is NOT used to re-sample
        demonstrations and does not guarantee bit-identical outputs.
        """
        raise NotImplementedError
