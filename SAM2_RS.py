"""
Import bridge for `SAM2-RS.py`.

The implementation file is named with a hyphen (`SAM2-RS.py`), which cannot be
imported as a standard Python module. This file provides a stable import path:

    from SAM2_RS import SAM2_RS
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_TARGET = _THIS_DIR / "SAM2-RS.py"

_spec = importlib.util.spec_from_file_location("sam2_rs_impl", _TARGET)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load model implementation from: {_TARGET}")

_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

SAM2_RS = _mod.SAM2_RS

__all__ = ["SAM2_RS"]

