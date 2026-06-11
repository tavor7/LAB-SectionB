"""Staged coordinate search for CE+RRF retrieve hyperparameters.

Uses median-fold NDCG@10 on public queries to reduce overfitting on small sets.
Does not write hparams.json unless --apply-best is passed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

_DEV_DIR = Path(__file__).resolve().parent
STUDENT_ROOT = _DEV_DIR.parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from config import load_hparams, retrieve_hparams_override, save_hparams
from eval import mean_ndcg_at_k
from main import run
from retrieve import clear_retrieve_cache
from utils import PUBLIC_QUERIES_PATH, load_public_queries, resolve_artifacts_dir

RESULTS_PATH = _DEV_DIR / "tune_ce_results.csv"


def _median_fold_ndcg(
    ranked: Sequence[Sequence[int]],
    ground_truth: Sequence[Set[int]],
    *,
    n_folds: int = 5,
) -> float:
    n = len(ranked)
    if n == 0:
        return 0.0
    n_folds = max(1, min(n_folds, n))
    fold_size = (n + n_folds - 1) // n_folds
    fold_means: List[float] = []
    for f in range(n_folds):
        start = f * fold_size
        end = min(start + fold_size, n)
        if start >= end:
            continue
        fold_means.append(mean_ndcg_at_k(ranked[start:end], ground_truth[start:end]))
    fold_means.sort()
    mid = len(fold_means) // 2
    if len(fold_means) % 2:
        return float(fold_means[mid])
    return float(0.5 * (fold_means[mid - 1] + fold_means[mid]))


def _eval_config(
    overrides: Dict[str, Any],
    queries: List[str],
    ground_truth: List[Set[int]],
    artifacts_dir: Path,
    *,
    clear_cache: bool = False,
) -> Tuple[float, float, float]:
    """Return (median_fold_ndcg, full_ndcg, elapsed_s)."""
    if clear_cache:
        clear_retrieve_cache()
    t0 = time.perf_counter()
    with retrieve_hparams_override(overrides):
        ranked = run(queries, artifacts_dir=artifacts_dir)
    elapsed = time.perf_counter() - t0
    fold_ndcg = _median_fold_ndcg(ranked, ground_truth)
    full_ndcg = mean_ndcg_at_k(ranked, ground_truth)
    return fold_ndcg, full_ndcg, elapsed


def _run_grid(
    base: Dict[str, Any],
    grid: Dict[str, List[Any]],
    queries: List[str],
    ground_truth: List[Set[int]],
    artifacts_dir: Path,
    *,
    stage: str,
    select_key: str = "fold",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    keys = list(grid.keys())
    rows: List[Dict[str, Any]] = []
    best_overrides = dict(base)
    best_score = -1.0

    for values in itertools.product(*(grid[k] for k in keys)):
        overrides = dict(base)
        for k, v in zip(keys, values):
            if k == "agg_pair":
                overrides["agg_max_weight"] = float(v[0])
                overrides["agg_mean_weight"] = float(v[1])
            else:
                overrides[k] = v
        fold_ndcg, full_ndcg, elapsed = _eval_config(overrides, queries, ground_truth, artifacts_dir)
        row_keys = {
            k: (f"{v[0]}/{v[1]}" if k == "agg_pair" else overrides.get(k, v))
            for k, v in zip(keys, values)
        }
        row = {
            "stage": stage,
            **row_keys,
            "median_fold_ndcg": fold_ndcg,
            "full_ndcg": full_ndcg,
            "elapsed_s": elapsed,
        }
        rows.append(row)
        score = fold_ndcg if select_key == "fold" else full_ndcg
        label = " ".join(f"{k}={overrides[k]}" for k in keys)
        print(
            f"  [{stage}] {label} -> fold={fold_ndcg:.4f} full={full_ndcg:.4f} t={elapsed:.1f}s",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_overrides = overrides

    print(f"[{stage}] best fold={best_score:.4f} -> {best_overrides}", flush=True)
    return best_overrides, rows


def _run_coordinate(
    base: Dict[str, Any],
    axes: Dict[str, List[Any]],
    queries: List[str],
    ground_truth: List[Set[int]],
    artifacts_dir: Path,
    *,
    stage: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Tune one hyperparameter at a time (coordinate ascent)."""
    current = dict(base)
    all_rows: List[Dict[str, Any]] = []

    for key, values in axes.items():
        best_local = dict(current)
        best_score = -1.0
        for v in values:
            overrides = dict(current)
            if key == "agg_pair":
                overrides["agg_max_weight"] = float(v[0])
                overrides["agg_mean_weight"] = float(v[1])
                label = f"agg_pair={v[0]}/{v[1]}"
            else:
                overrides[key] = v
                label = f"{key}={v}"
            fold_ndcg, full_ndcg, elapsed = _eval_config(
                overrides, queries, ground_truth, artifacts_dir
            )
            row = {
                "stage": f"{stage}:{key}",
                "param": key,
                "value": f"{v[0]}/{v[1]}" if key == "agg_pair" else v,
                "median_fold_ndcg": fold_ndcg,
                "full_ndcg": full_ndcg,
                "elapsed_s": elapsed,
            }
            all_rows.append(row)
            print(
                f"  [{stage}:{key}] {label} -> fold={fold_ndcg:.4f} full={full_ndcg:.4f} t={elapsed:.1f}s",
                flush=True,
            )
            if fold_ndcg > best_score:
                best_score = fold_ndcg
                best_local = overrides
        current = best_local
        print(f"[{stage}:{key}] best fold={best_score:.4f}", flush=True)

    return current, all_rows


def _append_csv(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    write_header = not RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune CE retrieve hyperparameters.")
    parser.add_argument("--artifacts-dir", default="", help="Artifact root")
    parser.add_argument("--stage", default="all", help="Stage name or 'all'")
    parser.add_argument("--apply-best", action="store_true", help="Write best to hparams.json")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    artifacts_dir = resolve_artifacts_dir(args.artifacts_dir or None)
    rows_data = load_public_queries()
    queries = [r["query"] for r in rows_data]
    ground_truth = [set(r["relevant_page_ids"]) for r in rows_data]

    hp = load_hparams()
    base = dict(hp.get("retrieve") or {})
    all_rows: List[Dict[str, Any]] = []

    print(f"artifacts={artifacts_dir} queries={len(queries)}", flush=True)
    clear_retrieve_cache()
    fold0, full0, t0 = _eval_config({}, queries, ground_truth, artifacts_dir)
    print(f"baseline fold={fold0:.4f} full={full0:.4f} t={t0:.1f}s", flush=True)

    grid_stages: Dict[str, Dict[str, List[Any]]] = {
        "ce_blend": {
            "cross_encoder_rrf_weight": [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0],
            "rerank_candidate_cap": [12, 16, 20, 24, 30],
        },
        "toggles": {
            "use_title_bm25": [True, False],
            "use_page_bm25": [True, False],
            "use_query_expansion": [True, False],
        },
    }
    coord_stages: Dict[str, Dict[str, List[Any]]] = {
        "rrf_weights": {
            "title_bm25_rrf_weight": [0.35, 0.5, 0.65, 0.8, 1.0],
            "page_bm25_rrf_weight": [0.0, 0.07, 0.35, 0.65, 0.86],
            "bm25_chunk_rrf_weight": [0.8, 1.0, 1.03, 1.2],
            "dense_rrf_weight": [0.8, 1.0, 1.2],
        },
        "recall": {
            "rrf_k": [10, 15, 20, 25, 30],
            "candidate_multiplier": [300, 400, 500, 600],
            "bm25_candidate_multiplier": [300, 400, 500, 600],
        },
        "aggregation": {
            "agg_pair": [(0.0, 1.0), (0.2, 0.8), (0.4, 0.6), (0.6, 0.4), (1.0, 0.0)],
        },
        "snippet": {
            "snippet_window_words": [80, 120, 160, 200],
            "snippet_step_words": [10, 20, 40],
        },
    }

    known = set(grid_stages) | set(coord_stages)
    run_stages = list(grid_stages.keys()) + list(coord_stages.keys()) if args.stage == "all" else [args.stage]
    current = dict(base)

    for stage_name in run_stages:
        if stage_name not in known:
            raise SystemExit(f"unknown stage: {stage_name}")
        print(f"\n=== stage {stage_name} ===", flush=True)
        if stage_name in grid_stages:
            current, stage_rows = _run_grid(
                current,
                grid_stages[stage_name],
                queries,
                ground_truth,
                artifacts_dir,
                stage=stage_name,
            )
        else:
            current, stage_rows = _run_coordinate(
                current,
                coord_stages[stage_name],
                queries,
                ground_truth,
                artifacts_dir,
                stage=stage_name,
            )
        all_rows.extend(stage_rows)

    _append_csv(all_rows)

    fold_f, full_f, tf = _eval_config(current, queries, ground_truth, artifacts_dir)
    print(
        f"\nfinal fold={fold_f:.4f} full={full_f:.4f} t={tf:.1f}s\n"
        f"best retrieve overrides:\n{json.dumps(current, indent=2)}",
        flush=True,
    )

    if args.apply_best:
        hp_out = copy.deepcopy(hp)
        retrieve = dict(hp_out.get("retrieve") or {})
        retrieve.update(current)
        hp_out["retrieve"] = retrieve
        save_hparams(hp_out)
        print(f"wrote {STUDENT_ROOT / 'hparams.json'}", flush=True)


if __name__ == "__main__":
    main()
