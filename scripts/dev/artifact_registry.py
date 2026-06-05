"""Manifest and paths for chunk-size sweep builds (local dev)."""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from index import (
    CHECKPOINT_NAME,
    META_NAME,
    PAGE_FEATURES_NAME,
    PAGE_IDS_NAME,
    _dense_artifacts_complete,
)
from lexical import has_bm25_index
from utils import ARTIFACTS_DIR, STUDENT_ROOT

ARTIFACTS_SWEEP_DIR = STUDENT_ROOT / "artifacts_sweep"
MANIFEST_NAME = "manifest.json"
RESULTS_NAME = "results.csv"

# Active sweep: build these only (coarse → finer). Overlap ≈ 25% of chunk_words.
DEFAULT_SWEEP_GRID: List[tuple[int, int]] = [
    (400, 100),
    (320, 80),
    (240, 60),
]

SWEEP_GRID_CHUNK_WORDS: List[int] = [400, 320, 240]

# w140_o35: legacy submission index (artifacts/) — register with `sweep register`, not rebuilt.
BASELINE_VARIANT_ID = "w140_o35"


def variant_id(chunk_words: int, overlap_words: int) -> str:
    return f"w{chunk_words}_o{overlap_words}"


def overlap_for_chunk_words(chunk_words: int) -> int:
    return max(1, round(chunk_words * 0.25))


def variant_dir(chunk_words: int, overlap_words: int, *, base: Optional[Path] = None) -> Path:
    root = base or ARTIFACTS_SWEEP_DIR
    return root / variant_id(chunk_words, overlap_words)


def manifest_path(base: Optional[Path] = None) -> Path:
    root = base or ARTIFACTS_SWEEP_DIR
    return root / MANIFEST_NAME


def load_manifest(base: Optional[Path] = None) -> Dict[str, Any]:
    path = manifest_path(base)
    if not path.is_file():
        return {"variants": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(data: Dict[str, Any], base: Optional[Path] = None) -> None:
    root = base or ARTIFACTS_SWEEP_DIR
    root.mkdir(parents=True, exist_ok=True)
    manifest_path(base).write_text(json.dumps(data, indent=2), encoding="utf-8")


def is_variant_complete(out_dir: Path) -> bool:
    if not _dense_artifacts_complete(out_dir):
        return False
    if not (out_dir / PAGE_FEATURES_NAME).is_file():
        return False
    if not has_bm25_index(out_dir, "chunk") and not (out_dir / "bm25_vocab.json").is_file():
        return False
    if not has_bm25_index(out_dir, "title"):
        return False
    if not has_bm25_index(out_dir, "page"):
        return False
    return True


def _variant_record(out_dir: Path, chunk_words: int, overlap_words: int) -> Dict[str, Any]:
    vid = variant_id(chunk_words, overlap_words)
    rel = os.path.relpath(out_dir.resolve(), STUDENT_ROOT)
    num_vectors: Optional[int] = None
    meta_path = out_dir / META_NAME
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        num_vectors = int(meta.get("num_vectors", 0)) or None
    elif (out_dir / PAGE_IDS_NAME).is_file():
        import numpy as np

        num_vectors = int(np.load(out_dir / PAGE_IDS_NAME).shape[0])

    return {
        "id": vid,
        "path": rel,
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "chunk_text_format": "title_content_v1",
        "num_vectors": num_vectors,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "complete": is_variant_complete(out_dir),
    }


def upsert_manifest_entry(
    out_dir: Path,
    chunk_words: int,
    overlap_words: int,
    base: Optional[Path] = None,
) -> Dict[str, Any]:
    data = load_manifest(base)
    rec = _variant_record(out_dir, chunk_words, overlap_words)
    variants: List[Dict[str, Any]] = [
        v for v in data.get("variants", []) if v.get("id") != rec["id"]
    ]
    variants.append(rec)
    variants.sort(key=lambda v: (v.get("chunk_words", 0), v.get("overlap_words", 0)))
    data["variants"] = variants
    save_manifest(data, base)
    return rec


def register_submission_baseline() -> Dict[str, Any]:
    """Register current artifacts/ as w140_o35 without copying."""
    return upsert_manifest_entry(ARTIFACTS_DIR, 140, 35)


def migrate_baseline_copy(*, force: bool = False) -> Path:
    """Copy submission artifacts/ into artifacts_sweep/w140_o35 if missing."""
    dest = variant_dir(140, 35)
    if dest.exists() and not force:
        if is_variant_complete(dest):
            upsert_manifest_entry(dest, 140, 35)
            return dest
    if not is_variant_complete(ARTIFACTS_DIR):
        raise FileNotFoundError(
            f"Submission artifacts incomplete under {ARTIFACTS_DIR}; "
            "run a full build first."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(ARTIFACTS_DIR, dest, ignore=shutil.ignore_patterns("shards"))
    ckpt = dest / CHECKPOINT_NAME
    if ckpt.is_file():
        ckpt.unlink()
    upsert_manifest_entry(dest, 140, 35)
    return dest


def resolve_variant_path(spec: str, *, base: Optional[Path] = None) -> Path:
    """Resolve variant id (w140_o35), relative path, or absolute path."""
    p = Path(spec)
    if p.is_absolute() and p.is_dir():
        return p.resolve()
    if p.is_dir():
        return p.resolve()
    sweep = base or ARTIFACTS_SWEEP_DIR
    candidate = sweep / spec
    if candidate.is_dir():
        return candidate.resolve()
    root = STUDENT_ROOT / spec
    if root.is_dir():
        return root.resolve()
    raise FileNotFoundError(f"Unknown artifact path or variant id: {spec!r}")


def list_complete_variants(base: Optional[Path] = None) -> List[Dict[str, Any]]:
    data = load_manifest(base)
    out: List[Dict[str, Any]] = []
    for v in data.get("variants", []):
        path = STUDENT_ROOT / v["path"]
        if path.is_dir() and is_variant_complete(path):
            entry = dict(v)
            entry["complete"] = True
            entry["resolved_path"] = str(path.resolve())
            out.append(entry)
    return out
