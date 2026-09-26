import json
import tempfile
from pathlib import Path

from main_eval.errors import MainEvalError
from main_eval.registry import load_registry, models_by_tier, resolve_model

REGISTRY = {
    "providers": {
        "openai": {"base_url": None, "api_key_env": "OPENAI_API_KEY"},
        "deepseek": {"base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY"},
    },
    "models": {
        "gpt-frontier": {"provider": "openai", "model": "gpt-4.1", "tier": "frontier"},
        "deepseek-chat": {"provider": "deepseek", "model": "deepseek-chat", "tier": "zh"},
        "r1": {
            "provider": "deepseek", "model": "deepseek-reasoner", "tier": "reasoning",
            "params": {"drop_temperature": True, "drop_seed": True},
        },
        "local-override": {
            "provider": "openai", "model": "custom", "base_url": "http://localhost:8000/v1",
            "api_key_env": "CUSTOM_KEY",
        },
    },
}


def _write(tmp, data):
    p = tmp / "registry.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_load_registry_missing():
    try:
        load_registry("does_not_exist.json")
    except MainEvalError:
        return
    raise AssertionError("expected MainEvalError for missing registry")


def test_load_registry_bad_json():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.json"
        p.write_text("{not json", encoding="utf-8")
        try:
            load_registry(p)
        except MainEvalError:
            return
        raise AssertionError("expected MainEvalError for bad JSON")


def test_resolve_alias_inherits_provider():
    resolved = resolve_model(REGISTRY, "gpt-frontier")
    assert resolved["model"] == "gpt-4.1"
    assert resolved["provider"] == "openai"
    assert resolved["base_url"] is None  # provider base_url is null
    assert resolved["api_key_env"] == "OPENAI_API_KEY"
    assert resolved["tier"] == "frontier"


def test_resolve_alias_with_base_url():
    resolved = resolve_model(REGISTRY, "deepseek-chat")
    assert resolved["base_url"] == "https://api.deepseek.com"
    assert resolved["api_key_env"] == "DEEPSEEK_API_KEY"


def test_resolve_params():
    resolved = resolve_model(REGISTRY, "r1")
    assert resolved["params"] == {"drop_temperature": True, "drop_seed": True}


def test_resolve_model_override():
    resolved = resolve_model(REGISTRY, "local-override")
    assert resolved["base_url"] == "http://localhost:8000/v1"
    assert resolved["api_key_env"] == "CUSTOM_KEY"


def test_resolve_raw_name_fallback():
    resolved = resolve_model(REGISTRY, "some-raw-model")
    assert resolved["model"] == "some-raw-model"
    assert resolved["provider"] is None
    assert resolved["base_url"] is None


def test_models_by_tier():
    assert models_by_tier(REGISTRY, "frontier") == ["gpt-frontier"]
    assert models_by_tier(REGISTRY, "reasoning") == ["r1"]
    assert models_by_tier(REGISTRY, "nope") == []


def test_real_registry_loads():
    here = Path(__file__).resolve().parents[1]
    registry = load_registry(here / "models" / "registry.json")
    assert "providers" in registry and "models" in registry
    for key, entry in registry["models"].items():
        assert "model" in entry and "provider" in entry
