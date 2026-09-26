"""Adapter registry / factory."""

from __future__ import annotations

from .base import BaseAdapter
from .mock import MockAdapter
from .openai_compat import OpenAICompatibleAdapter

__all__ = ["BaseAdapter", "MockAdapter", "OpenAICompatibleAdapter", "build_adapter"]


def build_adapter(
    name: str,
    *,
    model: str,
    api_key: str,
    base_url: str | None = None,
    params: dict | None = None,
) -> BaseAdapter:
    if name == "mock":
        return MockAdapter()
    if name == "openai_compat":
        return OpenAICompatibleAdapter(model=model, api_key=api_key, base_url=base_url, params=params)
    raise ValueError(f"unknown adapter {name!r}; expected 'mock' or 'openai_compat'")
