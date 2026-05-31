"""Offline index build and load (not timed at grading)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np

from chunk import Chunk, chunk_corpus
from config import hp_get, load_hparams
from embed import embed_texts
from lexical import (
    build_bm25_index,
    copy_bm25_artifacts,
    has_bm25_index,
    load_bm25_index,
    log_build_progress,
    save_bm25_index,
    _progress_step,
)
from utils import (
    ARTIFACTS_DIR,
    ensure_artifacts_dir,
    list_entry_paths,
    load_public_queries,
)

FAISS_INDEX_NAME = "faiss.index"
PAGE_IDS_NAME = "page_ids.npy"
META_NAME = "meta.json"
VECTORS_NAME = "index_vectors.npy"
SHARDS_DIR_NAME = "shards"
CHECKPOINT_NAME = "build_checkpoint.json"
PAGE_FEATURES_NAME = "page_features.npz"


def format_page_text(title: str, content: str) -> str:
    return f"Title: {title}\nContent: {content}"


def _as_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float32:
        return x.astype(np.float32, copy=False)
    return x


def _collect_chunk_texts(selected_paths: List[Path]) -> Tuple[List[str], np.ndarray]:
    """Re-read corpus pages and produce chunk texts aligned with dense index."""
    texts: List[str] = []
    page_ids_list: List[int] = []
    total = len(selected_paths)
    step = _progress_step(total)
    log_build_progress(0, total, "load corpus chunks")
    for i, path in enumerate(selected_paths):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["page_id"] = int(data.get("page_id", path.stem))
        for chunk in chunk_corpus([data]):
            texts.append(chunk.text)
            page_ids_list.append(chunk.page_id)
        if i % step == 0 or i == total - 1:
            log_build_progress(i + 1, total, "load corpus chunks")
    return texts, np.asarray(page_ids_list, dtype=np.int64)


def _collect_page_level_texts(
    selected_paths: List[Path],
) -> Tuple[List[str], List[str], np.ndarray]:
    """Title-only and full-page texts plus page_ids for page-level BM25."""
    title_texts: List[str] = []
    page_texts: List[str] = []
    page_ids_list: List[int] = []

    total = len(selected_paths)
    step = _progress_step(total)
    log_build_progress(0, total, "load corpus pages")
    for i, path in enumerate(selected_paths):
        data = json.loads(path.read_text(encoding="utf-8"))
        pid = int(data.get("page_id", path.stem))
        title = str(data.get("title", "")).strip()
        content = str(data.get("content", "")).strip()
        page_ids_list.append(pid)
        title_texts.append(format_page_text(title, title))
        page_texts.append(format_page_text(title, content))
        if i % step == 0 or i == total - 1:
            log_build_progress(i + 1, total, "load corpus pages")

    return title_texts, page_texts, np.asarray(page_ids_list, dtype=np.int64)


def _save_page_features(
    selected_paths: List[Path],
    out_dir: Path,
) -> None:
    page_ids_list: List[int] = []
    titles: List[str] = []
    contents: List[str] = []

    total = len(selected_paths)
    step = _progress_step(total)
    log_build_progress(0, total, "page_features load")
    for i, path in enumerate(selected_paths):
        data = json.loads(path.read_text(encoding="utf-8"))
        pid = int(data.get("page_id", path.stem))
        page_ids_list.append(pid)
        titles.append(str(data.get("title", "")).strip())
        contents.append(str(data.get("content", "")).strip())
        if i % step == 0 or i == total - 1:
            log_build_progress(i + 1, total, "page_features load")

    order = np.argsort(np.asarray(page_ids_list, dtype=np.int64))
    log_build_progress(total, total, "page_features save")
    pids = np.asarray(page_ids_list, dtype=np.int64)[order]
    np.savez_compressed(
        out_dir / PAGE_FEATURES_NAME,
        page_ids=pids,
        titles=np.asarray([titles[i] for i in order], dtype=object),
        contents=np.asarray([contents[i] for i in order], dtype=object),
    )


def _build_extended_artifacts(
    out_dir: Path,
    selected_paths: List[Path],
    hp: Dict[str, Any],
    *,
    chunk_bm25_index=None,
) -> None:
    """Title/page BM25, chunk BM25 aliases, and page_features.npz."""
    k1 = float(hp_get(hp, "bm25.k1", 1.5))
    b = float(hp_get(hp, "bm25.b", 0.75))

    title_texts, page_texts, page_pids = _collect_page_level_texts(selected_paths)

    print("[build] bm25_title=building...", flush=True)
    title_index = build_bm25_index(
        title_texts, page_pids, k1=k1, b=b, progress_label="bm25_title"
    )
    save_bm25_index(title_index, out_dir, prefix="title")
    print(f"[build] bm25_title=done n_docs={title_index.n_docs}", flush=True)

    print("[build] bm25_page=building...", flush=True)
    page_index = build_bm25_index(
        page_texts, page_pids, k1=k1, b=b, progress_label="bm25_page"
    )
    save_bm25_index(page_index, out_dir, prefix="page")
    print(f"[build] bm25_page=done n_docs={page_index.n_docs}", flush=True)

    if chunk_bm25_index is not None:
        save_bm25_index(chunk_bm25_index, out_dir, prefix="chunk")
        save_bm25_index(chunk_bm25_index, out_dir, prefix=None)
    elif has_bm25_index(out_dir, "chunk"):
        save_bm25_index(load_bm25_index(out_dir, prefix="chunk"), out_dir, prefix=None)
    elif (out_dir / "bm25_vocab.json").is_file():
        copy_bm25_artifacts(out_dir, src_prefix=None, dst_prefix="chunk")
        print("[build] bm25_chunk=aliased from legacy bm25_*", flush=True)
    else:
        print("[build] bm25_chunk=building from chunks...", flush=True)
        bm25_texts, bm25_page_ids = _collect_chunk_texts(selected_paths)
        chunk_index = build_bm25_index(
            bm25_texts, bm25_page_ids, k1=k1, b=b, progress_label="bm25_chunk"
        )
        save_bm25_index(chunk_index, out_dir, prefix="chunk")
        save_bm25_index(chunk_index, out_dir, prefix=None)
        print(f"[build] bm25_chunk=done n_docs={chunk_index.n_docs}", flush=True)

    print("[build] page_features=saving...", flush=True)
    _save_page_features(selected_paths, out_dir)
    print("[build] page_features=done", flush=True)


def _dense_artifacts_complete(out_dir: Path) -> bool:
    return (out_dir / FAISS_INDEX_NAME).is_file() and (out_dir / PAGE_IDS_NAME).is_file()


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Embed the full corpus and persist artifacts.

    Returns (vectors, page_ids) where row i corresponds to page_ids[i].
    For multi-chunk pipelines, store chunk metadata in index_meta.json and
    aggregate to page_id in retrieve.py.
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()
    hp = load_hparams()

    paths = list_entry_paths(entries_dir)
    shards_dir = out_dir / SHARDS_DIR_NAME
    shards_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / CHECKPOINT_NAME

    # Dev mode: build on a query-aware mini-corpus so public NDCG is meaningful.
    # - BUILD_DEV_PUBLIC=1: include all positive pages from public queries
    # - BUILD_DEV_NUM_QUERIES=10: only use first N public queries (default: all)
    # - BUILD_DEV_QUERY_SEED=S: if set, sample N queries with this seed
    # - BUILD_DEV_NEG_PAGES=N: include N additional negative pages (default 5000)
    # - BUILD_DEV_SEED=S: deterministic negative sampling (default 13)
    dev_public = os.environ.get("BUILD_DEV_PUBLIC", "").strip() in {"1", "true", "True"}
    dev_neg_env = os.environ.get("BUILD_DEV_NEG_PAGES", "").strip()
    dev_neg_pages = (
        int(dev_neg_env) if dev_neg_env else int(hp_get(hp, "dev_build.neg_pages", 5000))
    )
    dev_seed_env = os.environ.get("BUILD_DEV_SEED", "").strip()
    dev_seed = int(dev_seed_env) if dev_seed_env else int(hp_get(hp, "seed", 13))
    dev_num_q_env = os.environ.get("BUILD_DEV_NUM_QUERIES", "").strip()
    dev_num_queries = (
        int(dev_num_q_env) if dev_num_q_env else int(hp_get(hp, "dev_build.num_queries", 0))
    )
    dev_q_seed_env = os.environ.get("BUILD_DEV_QUERY_SEED", "").strip()
    dev_query_seed = (
        int(dev_q_seed_env)
        if dev_q_seed_env
        else int(hp_get(hp, "dev_build.query_seed", 0))
    )

    # Fast sanity-check mode: build on only the first N pages.
    # Set env var BUILD_SAMPLE_PAGES=N (e.g. 200) when iterating locally.
    sample_pages_env = os.environ.get("BUILD_SAMPLE_PAGES", "").strip()
    sample_pages = int(sample_pages_env) if sample_pages_env else 0

    records: List[Dict[str, Any]] = []
    selected_paths: List[Path]

    if dev_public:
        rows = load_public_queries()
        if dev_num_queries and dev_num_queries < len(rows):
            if dev_query_seed:
                rngq = np.random.default_rng(dev_query_seed)
                idx = rngq.choice(len(rows), size=dev_num_queries, replace=False)
                rows = [rows[int(i)] for i in idx]
            else:
                rows = rows[:dev_num_queries]

        positives: set[int] = set()
        for r in rows:
            positives.update(int(x) for x in r["relevant_page_ids"])

        pos_paths: List[Path] = []
        neg_paths: List[Path] = []
        for p in paths:
            try:
                pid = int(p.stem)
            except ValueError:
                neg_paths.append(p)
                continue
            if pid in positives:
                pos_paths.append(p)
            else:
                neg_paths.append(p)

        rng = np.random.default_rng(dev_seed)
        if dev_neg_pages > 0 and len(neg_paths) > dev_neg_pages:
            idx = rng.choice(len(neg_paths), size=dev_neg_pages, replace=False)
            sampled_negs = [neg_paths[int(i)] for i in idx]
        else:
            sampled_negs = neg_paths

        selected_paths = sorted(pos_paths) + sorted(sampled_negs)
        print(
            f"[build] dev_public=1 queries={len(rows)} positives={len(pos_paths)} negatives={len(sampled_negs)} total_pages={len(selected_paths)}",
            flush=True,
        )
    elif sample_pages:
        selected_paths = paths[:sample_pages]
        print(f"[build] sample_mode pages={sample_pages}", flush=True)
    else:
        selected_paths = paths

    # ---------- Checkpointed embedding build ----------
    pages_per_shard = int(hp_get(hp, "build.checkpoint_pages_per_shard", 200))
    embed_batch_size = int(hp_get(hp, "build.embed_batch_size", 64))
    pages_per_shard = max(1, pages_per_shard)
    embed_batch_size = max(1, embed_batch_size)

    # Resume if checkpoint exists and matches current selection + chunking params.
    selection_key = {
        "num_pages": int(len(selected_paths)),
        "first": str(selected_paths[0].name) if selected_paths else None,
        "last": str(selected_paths[-1].name) if selected_paths else None,
        "chunk_words": int(hp_get(hp, "chunking.chunk_words", 140)),
        "overlap_words": int(hp_get(hp, "chunking.overlap_words", 35)),
        "title_chunk": bool(hp_get(hp, "chunking.title_chunk", True)),
        "dev_public": bool(dev_public),
        "sample_pages": int(sample_pages),
    }
    start_i = 0
    shard_idx = 0
    resume_ok = False
    if checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if ckpt.get("selection_key") == selection_key:
                start_i = int(ckpt.get("next_page_index", 0))
                shard_idx = int(ckpt.get("next_shard_index", 0))
                resume_ok = True
                print(
                    f"[build] resume checkpoint next_page_index={start_i} next_shard_index={shard_idx}",
                    flush=True,
                )
            else:
                print(
                    "[build] checkpoint mismatch; clearing shards and restarting",
                    flush=True,
                )
        except Exception:
            print("[build] failed to read checkpoint; starting from scratch", flush=True)

    if not resume_ok:
        for old in shards_dir.glob("shard_*.npz"):
            old.unlink()
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    total_pages = len(selected_paths)
    skip_dense_build = (
        start_i >= total_pages
        and _dense_artifacts_complete(out_dir)
        and resume_ok
    )
    if skip_dense_build:
        print(
            "[build] dense artifacts present; skipping re-embed and FAISS rebuild",
            flush=True,
        )
    elif start_i >= total_pages:
        print("[build] embeddings already completed (checkpoint at end)", flush=True)
    else:
        for batch_start in range(start_i, total_pages, pages_per_shard):
            batch_paths = selected_paths[batch_start : batch_start + pages_per_shard]
            print(
                f"[build] shard={shard_idx} pages {batch_start+1}-{batch_start+len(batch_paths)}/{total_pages}",
                flush=True,
            )

            batch_records: List[Dict[str, Any]] = []
            for p in batch_paths:
                data = json.loads(p.read_text(encoding="utf-8"))
                data["page_id"] = int(data.get("page_id", p.stem))
                batch_records.append(data)

            batch_chunks: List[Chunk] = chunk_corpus(batch_records)
            texts = [c.text for c in batch_chunks]
            if not texts:
                shard_idx += 1
                continue

            vectors = _as_float32(
                embed_texts(
                    texts,
                    batch_size=embed_batch_size,
                    show_progress_bar=True,
                )
            )
            page_ids_arr = np.asarray([c.page_id for c in batch_chunks], dtype=np.int64)
            chunk_ids_arr = np.asarray([c.chunk_id for c in batch_chunks], dtype=np.int32)

            shard_path = shards_dir / f"shard_{shard_idx:05d}.npz"
            np.savez_compressed(
                shard_path,
                vectors=vectors,
                page_ids=page_ids_arr,
                chunk_ids=chunk_ids_arr,
            )

            # Update checkpoint after every shard.
            next_page_index = batch_start + len(batch_paths)
            shard_idx += 1
            checkpoint = {
                "selection_key": selection_key,
                "next_page_index": int(next_page_index),
                "next_shard_index": int(shard_idx),
            }
            checkpoint_path.write_text(
                json.dumps(checkpoint, indent=2), encoding="utf-8"
            )

    chunk_bm25_index = None
    M = int(hp_get(hp, "faiss_hnsw.M", 32))
    ef_construction = int(hp_get(hp, "faiss_hnsw.ef_construction", 200))
    save_vectors = bool(hp_get(hp, "retrieve.save_vectors", True))
    retrieve_mode = str(hp_get(hp, "retrieve.mode", "brute")).lower()

    if skip_dense_build:
        page_ids_arr = np.load(out_dir / PAGE_IDS_NAME)
        page_ids = [int(x) for x in page_ids_arr.tolist()]
        meta_path = out_dir / META_NAME
        if meta_path.exists():
            meta_prev = json.loads(meta_path.read_text(encoding="utf-8"))
            dim = int(meta_prev.get("embedding_dim", 384))
            chunk_ids_list = meta_prev.get("chunk_ids", [])
        else:
            dim = 384
            chunk_ids_list = []
        vec_path = out_dir / VECTORS_NAME
        vectors = np.load(vec_path, mmap_mode="r") if vec_path.exists() else np.zeros((0, dim))
    else:
        shard_files = sorted(shards_dir.glob("shard_*.npz"))
        if not shard_files:
            raise RuntimeError(
                f"No shard files found under {shards_dir}. Build cannot continue."
            )

        all_vecs: List[np.ndarray] = []
        all_page_ids: List[np.ndarray] = []
        all_chunk_ids: List[np.ndarray] = []
        for sf in shard_files:
            data = np.load(sf)
            all_vecs.append(_as_float32(data["vectors"]))
            all_page_ids.append(np.asarray(data["page_ids"], dtype=np.int64))
            all_chunk_ids.append(np.asarray(data["chunk_ids"], dtype=np.int32))

        vectors = np.vstack(all_vecs) if len(all_vecs) > 1 else all_vecs[0]
        page_ids_arr = (
            np.concatenate(all_page_ids) if len(all_page_ids) > 1 else all_page_ids[0]
        )
        chunk_ids_arr = (
            np.concatenate(all_chunk_ids)
            if len(all_chunk_ids) > 1
            else all_chunk_ids[0]
        )
        page_ids = [int(x) for x in page_ids_arr.tolist()]
        chunk_ids_list = [int(x) for x in chunk_ids_arr.tolist()]

        if vectors.ndim != 2 or vectors.shape[0] != len(page_ids):
            raise ValueError(
                f"Bad embedding matrix shape={vectors.shape}, num_page_ids={len(page_ids)}"
            )

        dim = int(vectors.shape[1])
        print(
            f"[build] faiss_build=indexhnswflat dim={dim} M={M} efC={ef_construction}",
            flush=True,
        )
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.add(vectors)

        print("[build] writing dense artifacts...", flush=True)
        faiss.write_index(index, str(out_dir / FAISS_INDEX_NAME))
        np.save(out_dir / PAGE_IDS_NAME, np.asarray(page_ids, dtype=np.int64))
        if save_vectors or retrieve_mode == "brute":
            np.save(out_dir / VECTORS_NAME, vectors)

        if not has_bm25_index(out_dir, "chunk") and not (
            out_dir / "bm25_vocab.json"
        ).is_file():
            print("[build] bm25_chunk=building...", flush=True)
            bm25_texts, bm25_page_ids = _collect_chunk_texts(selected_paths)
            if len(bm25_texts) != len(page_ids):
                raise ValueError(
                    f"BM25 chunk count {len(bm25_texts)} != dense count {len(page_ids)}"
                )
            if not np.array_equal(bm25_page_ids, page_ids_arr):
                raise ValueError("BM25 page_ids do not align with dense index page_ids")
            k1 = float(hp_get(hp, "bm25.k1", 1.5))
            b = float(hp_get(hp, "bm25.b", 0.75))
            chunk_bm25_index = build_bm25_index(
                bm25_texts,
                bm25_page_ids,
                k1=k1,
                b=b,
                progress_label="bm25_chunk",
            )
            save_bm25_index(chunk_bm25_index, out_dir, prefix="chunk")
            save_bm25_index(chunk_bm25_index, out_dir, prefix=None)
            print(f"[build] bm25_chunk=done n_docs={chunk_bm25_index.n_docs}", flush=True)

    _build_extended_artifacts(
        out_dir, selected_paths, hp, chunk_bm25_index=chunk_bm25_index
    )

    if skip_dense_build and (out_dir / META_NAME).exists():
        dim = int(
            json.loads((out_dir / META_NAME).read_text(encoding="utf-8")).get(
                "embedding_dim", 384
            )
        )
    elif not skip_dense_build:
        dim = int(vectors.shape[1])

    meta: Dict[str, Any] = {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": dim,
        "num_vectors": int(len(page_ids)),
        "index_type": "IndexHNSWFlat",
        "metric": "inner_product",
        "hnsw_M": M,
        "hnsw_ef_construction": ef_construction,
        "chunking": {
            "strategy": "word_windows",
            "chunk_words": int(hp_get(hp, "chunking.chunk_words", 140)),
            "overlap_words": int(hp_get(hp, "chunking.overlap_words", 35)),
            "title_chunk": bool(hp_get(hp, "chunking.title_chunk", True)),
        },
        "chunk_ids": chunk_ids_list,
        "has_vectors_npy": (out_dir / VECTORS_NAME).is_file(),
        "has_bm25": True,
        "has_bm25_chunk": has_bm25_index(out_dir, "chunk"),
        "has_bm25_title": has_bm25_index(out_dir, "title"),
        "has_bm25_page": has_bm25_index(out_dir, "page"),
        "has_page_features": (out_dir / PAGE_FEATURES_NAME).is_file(),
        "checkpointing": {
            "shards_dir": SHARDS_DIR_NAME,
            "checkpoint_file": CHECKPOINT_NAME,
            "pages_per_shard": pages_per_shard,
        },
    }
    if skip_dense_build and (out_dir / META_NAME).exists():
        prev = json.loads((out_dir / META_NAME).read_text(encoding="utf-8"))
        if not meta["chunk_ids"] and prev.get("chunk_ids"):
            meta["chunk_ids"] = prev["chunk_ids"]

    (out_dir / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[build] done artifacts_dir={out_dir}", flush=True)
    if isinstance(vectors, np.ndarray) and vectors.ndim == 2:
        return vectors, page_ids
    vec_path = out_dir / VECTORS_NAME
    if vec_path.exists():
        return np.load(vec_path), page_ids
    return np.zeros((len(page_ids), meta["embedding_dim"]), dtype=np.float32), page_ids


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[faiss.Index, np.ndarray, Dict[str, Any]]:
    """Load FAISS index + page_ids mapping from artifacts/."""
    root = artifacts_dir or ARTIFACTS_DIR
    index = faiss.read_index(str(root / FAISS_INDEX_NAME))
    page_ids = np.load(root / PAGE_IDS_NAME)
    meta = json.loads((root / META_NAME).read_text(encoding="utf-8"))
    return index, page_ids, meta
