"""Build and evaluate chunk-size variants under artifacts_sweep/."""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Sequence, Set, Tuple

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from artifact_registry import (
    ARTIFACTS_SWEEP_DIR,
    DEFAULT_SWEEP_GRID,
    RESULTS_NAME,
    is_variant_complete,
    list_complete_variants,
    load_manifest,
    migrate_baseline_copy,
    overlap_for_chunk_words,
    register_submission_baseline,
    resolve_variant_path,
    upsert_manifest_entry,
    variant_dir,
    variant_id,
)
from config import apply_sweep_retrieve_hparams, patch_chunking_hparams
from eval import mean_ndcg_at_k, ndcg_at_k
from index import build_index
from main import run
from retrieve import clear_retrieve_cache
from utils import ARTIFACTS_DIR, PUBLIC_QUERIES_PATH, load_public_queries


def _median_fold_ndcg(
    ranked: Sequence[Sequence[int]],
    ground_truth: Sequence[Set[int]],
    *,
    n_folds: int,
) -> Tuple[float, List[float]]:
    n = len(ranked)
    if n == 0:
        return 0.0, []
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
    sorted_folds = sorted(fold_means)
    mid = len(sorted_folds) // 2
    if not sorted_folds:
        return 0.0, []
    if len(sorted_folds) % 2:
        return sorted_folds[mid], fold_means
    return 0.5 * (sorted_folds[mid - 1] + sorted_folds[mid]), fold_means


def _fold_scores(
    queries: Sequence[str],
    ground_truth: Sequence[Set[int]],
    run_fn: Callable[[List[str]], List[List[int]]],
    *,
    n_folds: int,
) -> Tuple[float, float, List[float]]:
    """Return (mean_ndcg, median_fold_ndcg, per_fold_means)."""
    ranked = run_fn(list(queries))
    mean_all = mean_ndcg_at_k(ranked, list(ground_truth))
    median_fold, fold_means = _median_fold_ndcg(ranked, ground_truth, n_folds=n_folds)
    return mean_all, median_fold, fold_means


def cmd_list(_: argparse.Namespace) -> None:
    data = load_manifest()
    print(f"manifest={ARTIFACTS_SWEEP_DIR / 'manifest.json'}")
    for v in data.get("variants", []):
        path = STUDENT_ROOT / v["path"]
        ok = path.is_dir() and is_variant_complete(path)
        status = "complete" if ok else "incomplete"
        nv = v.get("num_vectors", "?")
        print(
            f"  {v.get('id')}: chunk={v.get('chunk_words')} overlap={v.get('overlap_words')} "
            f"vectors={nv} [{status}] -> {v.get('path')}"
        )
    complete = list_complete_variants()
    print(f"ready_for_eval={len(complete)}")


def cmd_register(_: argparse.Namespace) -> None:
    rec = register_submission_baseline()
    print(f"registered artifacts/ as {rec['id']} complete={rec['complete']}")


def cmd_migrate(args: argparse.Namespace) -> None:
    dest = migrate_baseline_copy(force=args.force)
    print(f"migrated baseline copy -> {dest}")


def cmd_build(args: argparse.Namespace) -> None:
    if args.chunk_words is not None:
        pairs = [(args.chunk_words, args.overlap or overlap_for_chunk_words(args.chunk_words))]
    elif args.all_grid:
        pairs = list(DEFAULT_SWEEP_GRID)
    else:
        raise SystemExit("build requires --chunk-words N or --all-grid")

    for cw, ov in pairs:
        vid = variant_id(cw, ov)
        out_dir = variant_dir(cw, ov)
        if is_variant_complete(out_dir) and not args.force:
            upsert_manifest_entry(out_dir, cw, ov)
            print(f"[build] skip {vid} (already complete)")
            continue

        print(f"[build] {vid} -> {out_dir}", flush=True)
        patch_chunking_hparams(chunk_words=cw, overlap_words=ov)
        out_dir.mkdir(parents=True, exist_ok=True)
        build_index(artifacts_dir=out_dir)
        rec = upsert_manifest_entry(out_dir, cw, ov)
        print(f"[build] done {vid} complete={rec['complete']} vectors={rec.get('num_vectors')}")


def cmd_eval(args: argparse.Namespace) -> None:
    from config import retrieve_hparams_override

    sweep_hp = apply_sweep_retrieve_hparams()
    rows = load_public_queries()
    queries = [r["query"] for r in rows]
    ground_truth = [set(r["relevant_page_ids"]) for r in rows]

    if args.variant:
        variants = [
            {
                "id": args.variant,
                "resolved_path": str(resolve_variant_path(args.variant)),
            }
        ]
    else:
        variants = list_complete_variants()
        if not variants:
            print("No complete variants in manifest. Run: build --all-grid or register/migrate.")
            return

    results_path = ARTIFACTS_SWEEP_DIR / RESULTS_NAME
    fieldnames = [
        "id",
        "chunk_words",
        "overlap_words",
        "mean_ndcg",
        "median_fold_ndcg",
        "query_phase_time",
        "num_vectors",
    ]
    out_rows: List[dict] = []

    with retrieve_hparams_override(sweep_hp):
        for v in variants:
            art_path = Path(v["resolved_path"])
            vid = v.get("id", art_path.name)
            print(f"[eval] {vid} path={art_path}", flush=True)

            clear_retrieve_cache()
            t0 = time.perf_counter()
            ranked = run(list(queries), artifacts_dir=art_path)
            elapsed = time.perf_counter() - t0
            mean_ndcg = mean_ndcg_at_k(ranked, list(ground_truth))
            median_fold, _ = _median_fold_ndcg(ranked, ground_truth, n_folds=args.folds)
            per_query = [
                ndcg_at_k(ranked[i], ground_truth[i]) for i in range(len(queries))
            ]

            row = {
                "id": vid,
                "chunk_words": v.get("chunk_words", ""),
                "overlap_words": v.get("overlap_words", ""),
                "mean_ndcg": f"{mean_ndcg:.4f}",
                "median_fold_ndcg": f"{median_fold:.4f}",
                "query_phase_time": f"{elapsed:.2f}",
                "num_vectors": v.get("num_vectors", ""),
            }
            out_rows.append(row)
            print(
                f"  mean_ndcg@10={mean_ndcg:.4f} median_fold={median_fold:.4f} "
                f"time={elapsed:.2f}s min_q={min(per_query):.3f} max_q={max(per_query):.3f}"
            )

    ARTIFACTS_SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {results_path}")

    if out_rows:
        best = max(out_rows, key=lambda r: float(r["median_fold_ndcg"]))
        print(
            f"best_by_median_fold: {best['id']} "
            f"median_fold={best['median_fold_ndcg']} mean={best['mean_ndcg']}"
        )


def cmd_ship(args: argparse.Namespace) -> None:
    apply_sweep_retrieve_hparams()
    results_path = ARTIFACTS_SWEEP_DIR / RESULTS_NAME
    if not results_path.is_file() and not args.variant:
        raise SystemExit(f"Missing {results_path}; run eval first or pass --variant")

    cw: int | None = None
    ov: int | None = None
    if args.variant:
        src = resolve_variant_path(args.variant)
        for v in load_manifest().get("variants", []):
            if v.get("id") == args.variant:
                cw = int(v["chunk_words"])
                ov = int(v["overlap_words"])
                break
    else:
        with results_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise SystemExit("results.csv is empty")
        best = max(rows, key=lambda r: float(r["median_fold_ndcg"]))
        vid = best["id"]
        src = resolve_variant_path(vid)
        cw = int(best["chunk_words"])
        ov = int(best["overlap_words"])
        print(f"shipping winner {vid} median_fold={best['median_fold_ndcg']}")

    src = Path(src).resolve()
    if not is_variant_complete(src):
        raise SystemExit(f"Incomplete artifacts: {src}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = STUDENT_ROOT / f"artifacts_backup_{stamp}"
    if ARTIFACTS_DIR.exists() and not args.no_backup:
        print(f"backup {ARTIFACTS_DIR} -> {backup}")
        shutil.move(str(ARTIFACTS_DIR), str(backup))

    print(f"copy {src} -> {ARTIFACTS_DIR}")
    shutil.copytree(src, ARTIFACTS_DIR, ignore=shutil.ignore_patterns("shards"))
    ckpt = ARTIFACTS_DIR / "build_checkpoint.json"
    if ckpt.is_file():
        ckpt.unlink()

    if cw is not None and ov is not None:
        patch_chunking_hparams(chunk_words=int(cw), overlap_words=int(ov))
        print(f"hparams chunking -> {cw}/{ov}")

    upsert_manifest_entry(ARTIFACTS_DIR, int(cw or 140), int(ov or 35))
    clear_retrieve_cache()
    print("done. Run: python scripts/eval_public.py")


def cmd_run_all(args: argparse.Namespace) -> None:
    """build --all-grid, eval, optional ship."""
    ns = argparse.Namespace(
        chunk_words=None,
        overlap=None,
        all_grid=True,
        force=args.force,
    )
    cmd_build(ns)
    ns_eval = argparse.Namespace(variant="", folds=args.folds)
    cmd_eval(ns_eval)
    if args.ship:
        cmd_ship(argparse.Namespace(variant="", no_backup=args.no_backup))


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk size sweep utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List manifest entries")
    p_list.set_defaults(func=cmd_list)

    p_reg = sub.add_parser("register", help="Register artifacts/ as w140_o35")
    p_reg.set_defaults(func=cmd_register)

    p_mig = sub.add_parser("migrate", help="Copy artifacts/ to artifacts_sweep/w140_o35")
    p_mig.add_argument("--force", action="store_true")
    p_mig.set_defaults(func=cmd_migrate)

    p_build = sub.add_parser("build", help="Build one or all grid variants")
    p_build.add_argument("--chunk-words", type=int, default=None)
    p_build.add_argument("--overlap", type=int, default=None)
    p_build.add_argument("--all-grid", action="store_true")
    p_build.add_argument("--force", action="store_true")
    p_build.set_defaults(func=cmd_build)

    p_eval = sub.add_parser("eval", help="Eval complete variants (fold NDCG)")
    p_eval.add_argument("--variant", type=str, default="", help="Single variant id or path")
    p_eval.add_argument("--folds", type=int, default=5)
    p_eval.set_defaults(func=cmd_eval)

    p_ship = sub.add_parser("ship", help="Copy eval winner (or --variant) into artifacts/")
    p_ship.add_argument("--variant", type=str, default="")
    p_ship.add_argument("--no-backup", action="store_true")
    p_ship.set_defaults(func=cmd_ship)

    p_all = sub.add_parser(
        "run-all",
        help="build --all-grid, eval all complete, optional --ship",
    )
    p_all.add_argument("--force", action="store_true")
    p_all.add_argument("--folds", type=int, default=5)
    p_all.add_argument("--ship", action="store_true")
    p_all.add_argument("--no-backup", action="store_true")
    p_all.set_defaults(func=cmd_run_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
