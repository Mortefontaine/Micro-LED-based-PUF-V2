"""Run dependency-light structural checks without requiring pytest."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "tests" / "test_release_smoke.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("release_smoke", TEST_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {TEST_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tests = sorted(name for name in dir(module) if name.startswith("test_"))
    for name in tests:
        getattr(module, name)()
        print(f"PASS {name}")
    print(f"{len(tests)} release checks passed")


if __name__ == "__main__":
    main()
