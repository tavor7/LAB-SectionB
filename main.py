"""
Section B entry point.

The autograder calls run(queries) once with all evaluation queries (batch of 50).
Query embedding + retrieval must complete within the time limit (GPU available).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from index import build_index
from retrieve import search_batch
from utils import resolve_artifacts_dir


def run(
    queries: List[str],
    *,
    artifacts_dir: Optional[Path | str] = None,
) -> List[List[int]]:
    """
    Rank corpus pages for each query.

    Parameters
    ----------
    queries : list[str]
        Batch of query strings (e.g. 50 hidden queries at grading time).
    artifacts_dir : path, optional
        Override artifact root (env ARTIFACTS_DIR also supported). Grading uses
        default artifacts/ when unset.

    Returns
    -------
    list[list[int]]
        One ranked list of page_id per query (most relevant first).
        Only the first 10 IDs per list are scored.
    """
    root = resolve_artifacts_dir(artifacts_dir) if artifacts_dir is not None else None
    return search_batch(queries, artifacts_dir=root)


def build_offline_index(*, artifacts_dir: Optional[Path | str] = None) -> None:
    """Run once locally to create artifacts/ (not timed at grading)."""
    root = resolve_artifacts_dir(artifacts_dir) if artifacts_dir is not None else None
    build_index(artifacts_dir=root)


if __name__ == "__main__":
    build_offline_index()
    print("Index built under artifacts/. Run: python scripts/eval_public.py")
