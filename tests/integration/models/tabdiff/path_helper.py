"""Put the TabDiff implementation under test on ``sys.path``."""

from __future__ import annotations

import sys
import types
from pathlib import Path


TABDIFF_DIR = Path(__file__).resolve().parents[4] / "src" / "midst_toolkit" / "models" / "tabdiff"


def _ensure_icecream_available() -> None:
    try:
        import icecream  # noqa: F401
    except ImportError:
        stub = types.ModuleType("icecream")
        stub.install = lambda: None  # type: ignore[attr-defined]
        sys.modules["icecream"] = stub


def load_implementation() -> Path:
    """Make ``tabdiff`` / ``src`` / ``utils_train`` importable from the toolkit tree.

    If a non-vendored ``tabdiff`` is already in ``sys.modules`` (pytest can bind the
    test package name when parent dirs lack ``__init__.py``), drop it so imports
    resolve to ``TABDIFF_DIR/tabdiff``.
    """
    _ensure_icecream_available()
    path = str(TABDIFF_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)

    vendored_pkg = (TABDIFF_DIR / "tabdiff").resolve()
    existing = sys.modules.get("tabdiff")
    if existing is not None:
        existing_paths = [Path(p).resolve() for p in getattr(existing, "__path__", [])]
        if vendored_pkg not in existing_paths:
            for key in list(sys.modules):
                if key == "tabdiff" or key.startswith("tabdiff."):
                    del sys.modules[key]
    return TABDIFF_DIR
