"""Fast dev evaluation on a small subset of public queries.

This file is optional; it exists to speed up iteration locally without
modifying the provided read-only scripts.
"""
from __future__ import annotations

import os
import time

from eval import evaluate_run
from main import run
from utils import load_public_queries


def main() -> None:
    rows = load_public_queries()
    n_env = os.environ.get("DEV_EVAL_NUM_QUERIES", "").strip()
    n = int(n_env) if n_env else 10
    rows = rows[:n]

    queries = [r["query"] for r in rows]
    ground_truth = [set(r["relevant_page_ids"]) for r in rows]

    t0 = time.perf_counter()
    stats = evaluate_run(queries, ground_truth, run)
    elapsed = time.perf_counter() - t0

    print(f"dev_public_queries={len(queries)}")
    print(f"mean_ndcg@10={stats['mean_ndcg@10']:.4f}")
    print(f"query_phase_time={elapsed:.2f}s")


if __name__ == "__main__":
    main()

