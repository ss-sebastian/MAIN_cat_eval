"""Standalone test runner (no pytest required).

Usage:  python tests/run_all.py
Also works under pytest (each test_* function is discovered normally).
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

MODULES = [
    "test_sampling",
    "test_validation",
    "test_child_separation",
    "test_prompt",
    "test_scores",
    "test_response",
    "test_metrics",
    "test_agreement",
    "test_registry",
    "test_e2e",
]


def main() -> int:
    total = 0
    failures = 0
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            total += 1
            try:
                fn()
                print(f"PASS {mod_name}.{name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {mod_name}.{name}: {exc}")
                traceback.print_exc()
    print(f"\n{total - failures}/{total} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
