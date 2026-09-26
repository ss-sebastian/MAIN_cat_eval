"""Model registry: resolve a model alias to a provider endpoint + API-key env var.

The registry is a plain JSON file (``models/registry.json``) so models can be
added or updated without touching code. Two sections:

- ``providers``: name -> {"base_url", "api_key_env"}. ``base_url`` may be null
  (use the OpenAI SDK default). ``api_key_env`` names the environment variable
  holding that provider's key (keys are never stored in the registry).
- ``models``: alias -> {"provider", "model", "tier"?, "params"?}. ``params`` is
  an optional per-model inference policy (e.g. ``drop_temperature`` for
  reasoning models that reject an explicit temperature).

Resolving a key that is *not* a registry alias falls back to treating it as a
raw model name, which keeps the original ``--model NAME`` usage working.
"""

from __future__ import annotations

import json
from pathlib import Path

from .errors import MainEvalError


class RegistryError(MainEvalError):
    """Raised for a missing or malformed model registry."""


def load_registry(path: str | Path) -> dict:
    """Read and parse a model registry JSON file."""
    path = Path(path)
    if not path.exists():
        raise RegistryError(f"model registry not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"model registry {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RegistryError(f"model registry {path} must be a JSON object")
    return data


def resolve_model(registry: dict, key: str) -> dict:
    """Resolve a model alias (or raw name) into a config dict.

    Returns keys: key, model, provider, base_url, api_key_env, params, tier.
    ``base_url``/``api_key_env`` fall back to the provider defaults; a model may
    override them. ``params`` and ``tier`` may be None/absent.
    """
    models = registry.get("models") or {}
    providers = registry.get("providers") or {}

    if key in models:
        entry = models[key]
        if not isinstance(entry, dict):
            raise RegistryError(f"registry model {key!r} must be an object")
        if "model" not in entry:
            raise RegistryError(f"registry model {key!r} is missing the 'model' field")
        provider_name = entry.get("provider")
        base_url = entry.get("base_url")
        api_key_env = entry.get("api_key_env")
        params = entry.get("params") or {}
        if provider_name and provider_name in providers:
            prov = providers[provider_name]
            if base_url is None:
                base_url = prov.get("base_url")
            if api_key_env is None:
                api_key_env = prov.get("api_key_env")
        return {
            "key": key,
            "model": entry["model"],
            "provider": provider_name,
            "base_url": base_url,
            "api_key_env": api_key_env,
            "params": params,
            "tier": entry.get("tier"),
        }

    # Fallback: treat the key as a raw model name (backward compatible).
    return {
        "key": key,
        "model": key,
        "provider": None,
        "base_url": None,
        "api_key_env": None,
        "params": {},
        "tier": None,
    }


def models_by_tier(registry: dict, tier: str) -> list[str]:
    """Return the aliases of all models in the given tier."""
    models = registry.get("models") or {}
    return [k for k, m in models.items() if isinstance(m, dict) and m.get("tier") == tier]
