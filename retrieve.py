"""Query-time hybrid retrieval (dense + BM25, RRF fusion)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import hp_get, load_hparams
from embed import embed_queries
from index import load_index
from lexical import load_bm25_index
from utils import ARTIFACTS_DIR, K_EVAL

_CACHED_PAGE_IDS = None
_CACHED_FAISS_INDEX = None
_CACHED_BM25_INDEX = None
_CACHED_VECTORS = None


def _get_page_ids(root: Path) -> np.ndarray:
    """Load and cache chunk row → page_id mapping."""
    global _CACHED_PAGE_IDS
    if _CACHED_PAGE_IDS is None:
        _CACHED_PAGE_IDS = np.load(root / "page_ids.npy")
    return _CACHED_PAGE_IDS


def _get_bm25_index(artifacts_dir: Optional[Path]):
    """Load and cache BM25 index from artifacts."""
    global _CACHED_BM25_INDEX
    if _CACHED_BM25_INDEX is None:
        _CACHED_BM25_INDEX = load_bm25_index(artifacts_dir)
    return _CACHED_BM25_INDEX


def _get_faiss_index(artifacts_dir: Optional[Path]):
    """Load and cache FAISS HNSW index from artifacts."""
    global _CACHED_FAISS_INDEX
    if _CACHED_FAISS_INDEX is None:
        _CACHED_FAISS_INDEX, _, _ = load_index(artifacts_dir)
    return _CACHED_FAISS_INDEX


def _get_vectors(root: Path) -> np.ndarray:
    """Memory-map dense embedding matrix for brute-force search."""
    global _CACHED_VECTORS
    if _CACHED_VECTORS is None:
        _CACHED_VECTORS = np.load(root / "index_vectors.npy", mmap_mode="r")
    return _CACHED_VECTORS


def _page_ranking_from_chunk_scores(
    chunk_indices: np.ndarray,
    chunk_scores: np.ndarray,
    page_ids: np.ndarray,
    *,
    agg: str = "max",
) -> List[int]:
    """Aggregate chunk-level scores into a ranked list of page_ids."""
    agg = agg.lower()
    page_chunks: Dict[int, List[float]] = {}

    for idx, score in zip(chunk_indices, chunk_scores):
        if idx < 0:
            continue
        pid = int(page_ids[int(idx)])
        page_chunks.setdefault(pid, []).append(float(score))

    page_scores: Dict[int, float] = {}
    for pid, scores in page_chunks.items():
        scores.sort(reverse=True)
        if agg == "sum":
            page_scores[pid] = float(sum(scores))
        elif agg == "mean_top3":
            page_scores[pid] = float(np.mean(scores[:3]))
        elif agg == "hybrid":
            page_scores[pid] = float(scores[0] + 0.2 * sum(scores[1:]))
        elif agg == "mean_top2":
            page_scores[pid] = float(np.mean(scores[:2]))
        elif agg == "mean_top4":
            page_scores[pid] = float(np.mean(scores[:4]))
        elif agg == "max_plus_mean_top3":
            page_scores[pid] = float(0.5 * scores[0] + 0.5 * np.mean(scores[:3]))
        else:
            page_scores[pid] = float(scores[0])

    items = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in items]


def _rrf_fuse(
    rankings: List[List[int]],
    *,
    rrf_k: int,
    top_k: int,
    weights: Optional[List[float]] = None,
) -> List[int]:
    """Weighted reciprocal-rank fusion over page rankings."""
    scores: Dict[int, float] = {}
    if weights is None:
        weights = [1.0] * len(rankings)

    for ranking, weight in zip(rankings, weights):
        for rank, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + weight / (rrf_k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in fused[:top_k]]


def _dense_brute_rankings(
    query_vectors: np.ndarray,
    page_ids: np.ndarray,
    vectors: np.ndarray,
    *,
    candidate_k: int,
    agg: str,
) -> List[List[int]]:
    """Top-k chunk search via matrix multiply, then page aggregation."""
    scores = query_vectors @ vectors.T
    out: List[List[int]] = []

    for i in range(len(query_vectors)):
        row = scores[i]
        k = min(candidate_k, len(row))

        if k >= len(row):
            order = np.argsort(-row)
        else:
            part = np.argpartition(-row, k - 1)[:k]
            order = part[np.argsort(-row[part])]

        out.append(
            _page_ranking_from_chunk_scores(
                order.astype(np.int64),
                row[order],
                page_ids,
                agg=agg,
            )
        )

    return out


def _dense_hnsw_rankings(
    query_vectors: np.ndarray,
    page_ids: np.ndarray,
    index,
    *,
    candidate_k: int,
    ef_min: int,
    ef_cap: int,
    agg: str,
) -> List[List[int]]:
    """FAISS HNSW chunk search, then page aggregation."""
    index.hnsw.efSearch = max(ef_min, min(candidate_k, ef_cap))
    distances, indices = index.search(query_vectors, candidate_k)
    out: List[List[int]] = []

    for i in range(len(query_vectors)):
        out.append(
            _page_ranking_from_chunk_scores(
                indices[i],
                distances[i],
                page_ids,
                agg=agg,
            )
        )

    return out


def _bm25_rankings(
    queries: List[str],
    bm25_index,
    *,
    candidate_k: int,
    agg: str,
) -> List[List[int]]:
    """BM25 chunk search per query, then page aggregation."""
    out: List[List[int]] = []

    for query in queries:
        doc_idx, doc_scores = bm25_index.search(query, top_k=candidate_k)
        out.append(
            _page_ranking_from_chunk_scores(
                doc_idx,
                doc_scores,
                bm25_index.page_ids,
                agg=agg,
            )
        )

    return out


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """
    Rank pages for each query using hybrid retrieval.

    Pipeline: embed queries → dense candidates (HNSW or brute) → optional BM25
    → weighted RRF at page level. Loads prebuilt artifacts only (no rebuild).
    """
    if not queries:
        return []

    hp = load_hparams()
    root = artifacts_dir or ARTIFACTS_DIR
    page_ids = _get_page_ids(root)
    query_vectors = embed_queries(queries)

    mult = int(hp_get(hp, "retrieve.candidate_multiplier", 50))
    candidate_k = max(top_k * max(1, mult), top_k)

    bm25_mult = int(hp_get(hp, "retrieve.bm25_candidate_multiplier", 50))
    bm25_candidate_k = max(top_k * max(1, bm25_mult), top_k)

    mode = str(hp_get(hp, "retrieve.mode", "brute")).lower()
    use_bm25 = bool(hp_get(hp, "retrieve.use_bm25", True))
    rrf_k = int(hp_get(hp, "retrieve.rrf_k", 60))
    ef_min = int(hp_get(hp, "faiss_hnsw.ef_search_min", 128))
    ef_cap = int(hp_get(hp, "faiss_hnsw.ef_search_cap", 256))
    agg = str(hp_get(hp, "retrieve.page_aggregation", "max"))

    dense_rrf_weight = float(hp_get(hp, "retrieve.dense_rrf_weight", 1.0))
    bm25_rrf_weight = float(hp_get(hp, "retrieve.bm25_rrf_weight", 1.0))

    if mode == "brute":
        vec_path = root / "index_vectors.npy"
        if not vec_path.exists():
            raise FileNotFoundError(
                f"Brute mode requires {vec_path}. Rebuild index with retrieve.mode=brute."
            )
        dense_rankings = _dense_brute_rankings(
            query_vectors,
            page_ids,
            _get_vectors(root),
            candidate_k=candidate_k,
            agg=agg,
        )
    else:
        index = _get_faiss_index(artifacts_dir)
        dense_rankings = _dense_hnsw_rankings(
            query_vectors,
            page_ids,
            index,
            candidate_k=candidate_k,
            ef_min=ef_min,
            ef_cap=ef_cap,
            agg=agg,
        )

    if not use_bm25:
        return [ranking[:top_k] for ranking in dense_rankings]

    bm25_index = _get_bm25_index(artifacts_dir)
    bm25_rankings = _bm25_rankings(
        queries,
        bm25_index,
        candidate_k=bm25_candidate_k,
        agg=agg,
    )

    ranked: List[List[int]] = []
    for i in range(len(queries)):
        fused = _rrf_fuse(
            [dense_rankings[i], bm25_rankings[i]],
            rrf_k=rrf_k,
            top_k=top_k,
            weights=[dense_rrf_weight, bm25_rrf_weight],
        )
        ranked.append(fused)

    return ranked
