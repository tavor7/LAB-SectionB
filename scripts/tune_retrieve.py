"""Grid-search retrieve hyperparameters on sweep artifact variants."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from artifact_registry import (
    ARTIFACTS_SWEEP_DIR,
    is_variant_complete,
    resolve_variant_path,
)
from config import SUBMISSION_RETRIEVE_BASELINE, retrieve_hparams_override
from eval import mean_ndcg_at_k
from main import run
from retrieve import clear_retrieve_cache
from utils import load_public_queries

TUNE_RESULTS_NAME = "tune_results.csv"


def _median_fold_ndcg(
    ranked: Sequence[Sequence[int]],
    ground_truth: Sequence[Set[int]],
    *,
    n_folds: int,
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
        fold_means.append(
            mean_ndcg_at_k(ranked[start:end], ground_truth[start:end])
        )
    if not fold_means:
        return 0.0
    fold_means.sort()
    mid = len(fold_means) // 2
    if len(fold_means) % 2:
        return float(fold_means[mid])
    return float(0.5 * (fold_means[mid - 1] + fold_means[mid]))


def _config_name(overrides: Dict[str, Any]) -> str:
    keys = (
        "candidate_multiplier",
        "rerank_candidate_cap",
        "expand_chunk_bm25",
        "title_bm25_rrf_weight",
        "page_bm25_rrf_weight",
        "bm25_chunk_rrf_weight",
        "features",
    )
    parts: List[str] = []
    for k in keys:
        if k not in overrides:
            continue
        v = overrides[k]
        if k == "expand_chunk_bm25":
            parts.append("ecx1" if v else "ecx0")
        elif k == "features":
            parts.append(str(v))
        elif isinstance(v, float):
            parts.append(f"{k[:3]}{v:.2f}".replace(".", ""))
        else:
            parts.append(f"{k[:3]}{v}")
    return "_".join(parts) if parts else "default"


def build_tune_grid() -> List[Tuple[str, Dict[str, Any]]]:
    """Focused grid: baseline anchor + cartesian on key retrieve knobs."""
    configs: List[Tuple[str, Dict[str, Any]]] = []

    # Anchor: submission-tuned baseline (~0.2546 on w140).
    configs.append(
        (
            "submission_baseline",
            dict(SUBMISSION_RETRIEVE_BASELINE),
        )
    )

    feature_modes = {
        "off": {
            "title_overlap_weight": 0.0,
            "page_coverage_weight": 0.0,
            "phrase_bonus_weight": 0.0,
        },
        "light": {
            "title_overlap_weight": 0.08,
            "page_coverage_weight": 0.1,
            "phrase_bonus_weight": 0.05,
        },
    }

    tune_keys: Dict[str, Any] = {}
    for cmult, rcap, ecx, w_title, w_page, feat_name in itertools.product(
        [400, 500],
        [30, 120],
        [False, True],
        [0.35, 0.6],
        [0.86, 1.0],
        ["off", "light"],
    ):
        base = dict(SUBMISSION_RETRIEVE_BASELINE)
        base.update(feature_modes[feat_name])
        base.update(
            {
                "candidate_multiplier": cmult,
                "bm25_candidate_multiplier": cmult,
                "rerank_candidate_cap": rcap,
                "expand_chunk_bm25": ecx,
                "title_bm25_rrf_weight": w_title,
                "page_bm25_rrf_weight": w_page,
                "features": feat_name,
            }
        )
        # retrieve.py ignores unknown keys; strip label before override.
        overrides = {k: v for k, v in base.items() if k != "features"}
        overrides["features"] = feat_name  # kept for CSV only
        name = _config_name(base)
        configs.append((name, overrides))

    # Deduplicate by JSON-serialized overrides (keep first name).
    seen: set[str] = set()
    unique: List[Tuple[str, Dict[str, Any]]] = []
    for name, ov in configs:
        key = json.dumps(ov, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, ov))
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune retrieve hparams on variants.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["w140_o35", "w400_o100"],
        help="Sweep variant ids or paths",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--max-configs",
        type=int,
        default=0,
        help="Limit grid size (0 = all)",
    )
    args = parser.parse_args()

    rows_pub = load_public_queries()
    queries = [r["query"] for r in rows_pub]
    ground_truth = [set(r["relevant_page_ids"]) for r in rows_pub]

    variants: List[Tuple[str, Path]] = []
    for spec in args.variants:
        path = resolve_variant_path(spec)
        if not is_variant_complete(path):
            print(f"skip incomplete variant: {spec} -> {path}")
            continue
        variants.append((Path(spec).name if "/" not in spec else path.name, path))
    if not variants:
        raise SystemExit("No complete variants to tune.")

    grid = build_tune_grid()
    if args.max_configs > 0:
        grid = grid[: args.max_configs]

    out_path = ARTIFACTS_SWEEP_DIR / TUNE_RESULTS_NAME
    ARTIFACTS_SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "variant",
        "config",
        "mean_ndcg",
        "median_fold_ndcg",
        "query_time_s",
        "candidate_multiplier",
        "rerank_candidate_cap",
        "expand_chunk_bm25",
        "title_bm25_rrf_weight",
        "page_bm25_rrf_weight",
        "bm25_chunk_rrf_weight",
        "features",
    ]
    results: List[Dict[str, Any]] = []
    total = len(variants) * len(grid)
    done = 0

    print(f"tuning variants={[v[0] for v in variants]} configs={len(grid)} total_runs={total}")

    for variant_name, art_path in variants:
        best_mean = -1.0
        best_row: Dict[str, Any] | None = None
        for config_name, overrides in grid:
            done += 1
            t0 = time.perf_counter()
            clear_retrieve_cache()
            hp_patch = {k: v for k, v in overrides.items() if k != "features"}
            with retrieve_hparams_override(hp_patch):
                ranked = run(list(queries), artifacts_dir=art_path)
            elapsed = time.perf_counter() - t0
            mean_ndcg = mean_ndcg_at_k(ranked, ground_truth)
            median_fold = _median_fold_ndcg(ranked, ground_truth, n_folds=args.folds)

            row = {
                "variant": variant_name,
                "config": config_name,
                "mean_ndcg": f"{mean_ndcg:.4f}",
                "median_fold_ndcg": f"{median_fold:.4f}",
                "query_time_s": f"{elapsed:.2f}",
                "candidate_multiplier": overrides.get("candidate_multiplier", ""),
                "rerank_candidate_cap": overrides.get("rerank_candidate_cap", ""),
                "expand_chunk_bm25": overrides.get("expand_chunk_bm25", ""),
                "title_bm25_rrf_weight": overrides.get("title_bm25_rrf_weight", ""),
                "page_bm25_rrf_weight": overrides.get("page_bm25_rrf_weight", ""),
                "bm25_chunk_rrf_weight": overrides.get("bm25_chunk_rrf_weight", ""),
                "features": overrides.get("features", ""),
            }
            results.append(row)
            if mean_ndcg > best_mean:
                best_mean = mean_ndcg
                best_row = row

            print(
                f"[{done}/{total}] {variant_name} {config_name}: "
                f"mean={mean_ndcg:.4f} fold={median_fold:.4f} t={elapsed:.1f}s",
                flush=True,
            )

        if best_row:
            print(
                f"BEST {variant_name}: {best_row['config']} "
                f"mean={best_row['mean_ndcg']} fold={best_row['median_fold_ndcg']}",
                flush=True,
            )

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {out_path}")

    # Global best by median fold (primary), then mean.
    best = max(
        results,
        key=lambda r: (float(r["median_fold_ndcg"]), float(r["mean_ndcg"])),
    )
    print(
        f"\nOVERALL BEST (median fold): variant={best['variant']} config={best['config']} "
        f"mean={best['mean_ndcg']} median_fold={best['median_fold_ndcg']}"
    )
    best_mean = max(results, key=lambda r: float(r["mean_ndcg"]))
    print(
        f"OVERALL BEST (mean): variant={best_mean['variant']} config={best_mean['config']} "
        f"mean={best_mean['mean_ndcg']} median_fold={best_mean['median_fold_ndcg']}"
    )


if __name__ == "__main__":
    main()
