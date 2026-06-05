"""Self-evaluation on the 50 public queries (mean NDCG@10).

Grading-style check: run from repo root after clone + git lfs pull, without
rebuilding the index. Requires data/public_queries.json and artifacts/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from eval import evaluate_run, load_query_file
from main import run
from retrieve import clear_retrieve_cache
from utils import PUBLIC_QUERIES_PATH, resolve_artifacts_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on public queries.")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="",
        help="Artifact root (default: artifacts/ or ARTIFACTS_DIR env)",
    )
    args = parser.parse_args()

    artifacts_dir = resolve_artifacts_dir(args.artifacts_dir or None)
    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    ground_truth = [r["relevant_page_ids"] for r in rows]

    def run_fn(q: list[str]) -> list[list[int]]:
        return run(q, artifacts_dir=artifacts_dir)

    clear_retrieve_cache()
    t0 = time.perf_counter()
    stats = evaluate_run(queries, ground_truth, run_fn)
    elapsed = time.perf_counter() - t0

    print(f"artifacts_dir={artifacts_dir}")
    print(f"public_queries={len(queries)}")
    print(f"mean_ndcg@10={stats['mean_ndcg@10']:.4f}")
    print(f"query_phase_time={elapsed:.2f}s")


if __name__ == "__main__":
    main()
