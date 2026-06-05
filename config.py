"""Hyperparameter loading (edit hparams.json for tuning)."""
from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from utils import STUDENT_ROOT

HPARAMS_PATH = STUDENT_ROOT / "hparams.json"

_RETRIEVE_OVERRIDE: Optional[Dict[str, Any]] = None

# Tuned retrieve block that scored ~0.2546 on legacy w140 artifacts/.
SUBMISSION_RETRIEVE_BASELINE: Dict[str, Any] = {
    "mode": "hnsw",
    "save_vectors": True,
    "use_bm25": True,
    "candidate_multiplier": 400,
    "bm25_candidate_multiplier": 400,
    "rerank_candidate_cap": 30,
    "rrf_score_cap": 500,
    "page_aggregation": "max_plus_mean_top3",
    "rrf_k": 15,
    "dense_rrf_weight": 1.0,
    "bm25_chunk_rrf_weight": 1.03,
    "title_bm25_rrf_weight": 0.35,
    "page_bm25_rrf_weight": 0.86,
    "title_overlap_weight": 0.02,
    "title_coverage_weight": 0,
    "page_coverage_weight": 0.0,
    "phrase_bonus_weight": 0,
    "use_query_expansion": True,
    "expand_chunk_bm25": False,
    "use_dual_query_rrf": False,
    "use_title_bm25": True,
    "use_page_bm25": True,
}


def load_hparams() -> Dict[str, Any]:
    """
    Load hyperparameters from hparams.json.
    The file is intentionally simple JSON so it's easy to tweak.
    """
    if not HPARAMS_PATH.exists():
        hp: Dict[str, Any] = {}
    else:
        hp = json.loads(HPARAMS_PATH.read_text(encoding="utf-8"))
    if _RETRIEVE_OVERRIDE is not None:
        hp = copy.deepcopy(hp)
        retrieve = dict(hp.get("retrieve") or {})
        retrieve.update(_RETRIEVE_OVERRIDE)
        hp["retrieve"] = retrieve
    return hp


@contextmanager
def retrieve_hparams_override(overrides: Dict[str, Any]) -> Iterator[None]:
    """Temporarily merge keys into retrieve.* for tuning (does not write disk)."""
    global _RETRIEVE_OVERRIDE
    prev = _RETRIEVE_OVERRIDE
    _RETRIEVE_OVERRIDE = dict(overrides)
    try:
        yield
    finally:
        _RETRIEVE_OVERRIDE = prev


def hp_get(hp: Dict[str, Any], path: str, default):
    cur: Any = hp
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def save_hparams(hp: Dict[str, Any]) -> None:
    HPARAMS_PATH.write_text(json.dumps(hp, indent=2) + "\n", encoding="utf-8")


SWEEP_RETRIEVE_OVERRIDES: Dict[str, Any] = {
    "candidate_multiplier": 500,
    "bm25_candidate_multiplier": 500,
    "rerank_candidate_cap": 120,
    "dense_rrf_weight": 1.0,
    "bm25_chunk_rrf_weight": 1.1,
    "title_bm25_rrf_weight": 0.6,
    "page_bm25_rrf_weight": 1.0,
    "title_overlap_weight": 0.08,
    "page_coverage_weight": 0.1,
    "phrase_bonus_weight": 0.05,
    "expand_chunk_bm25": True,
}


def apply_sweep_retrieve_hparams() -> Dict[str, Any]:
    """Return sweep retrieve settings (in-memory only; does not write hparams.json)."""
    return dict(SWEEP_RETRIEVE_OVERRIDES)


def patch_chunking_hparams(
    *,
    chunk_words: int,
    overlap_words: int,
    title_chunk: bool = True,
) -> Dict[str, Any]:
    """Update hparams.json chunking block and return full hparams dict."""
    hp = load_hparams()
    chunking = dict(hp.get("chunking") or {})
    chunking["chunk_words"] = int(chunk_words)
    chunking["overlap_words"] = int(overlap_words)
    chunking["title_chunk"] = bool(title_chunk)
    hp["chunking"] = chunking
    save_hparams(hp)
    return hp

