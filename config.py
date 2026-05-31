"""Hyperparameter loading (edit hparams.json for tuning)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from utils import STUDENT_ROOT

HPARAMS_PATH = STUDENT_ROOT / "hparams.json"


def load_hparams() -> Dict[str, Any]:
    """
    Load hyperparameters from hparams.json.
    The file is intentionally simple JSON so it's easy to tweak.
    """
    if not HPARAMS_PATH.exists():
        return {}
    return json.loads(HPARAMS_PATH.read_text(encoding="utf-8"))


def hp_get(hp: Dict[str, Any], path: str, default):
    cur: Any = hp
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

