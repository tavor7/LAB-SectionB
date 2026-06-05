
"""Query-time hybrid retrieval (Hyper-optimized Cross-Encoder for CPU Execution)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import hp_get, load_hparams
from embed import embed_queries
from index import load_index
from lexical import has_bm25_index, load_bm25_index, tokenize
from query_expand import query_versions
from utils import ARTIFACTS_DIR, K_EVAL, resolve_artifacts_dir

try:
    from sentence_transformers import CrossEncoder
    import torch
    torch.set_num_threads(min(os.cpu_count() or 4, 8))
except ImportError:  # pragma: no cover
    CrossEncoder = None  # type: ignore

PAGE_FEATURES_NAME = "page_features.npz"

_CACHED_PAGE_IDS = None
_CACHED_FAISS_INDEX = None
_CACHED_VECTORS = None
_CACHED_PAGE_DENSE_PAGE_IDS = None
_CACHED_PAGE_DENSE_VECTORS = None
_CACHED_BM25_CHUNK = None
_CACHED_BM25_TITLE = None
_CACHED_BM25_PAGE = None
_CACHED_CROSS_ENCODER = None
_CACHED_PAGE_LOOKUP: Optional[Dict[int, Tuple[str, str]]] = None


def clear_retrieve_cache() -> None:
    """Reset loaded indexes (needed when switching artifacts_dir in one process)."""
    global _CACHED_PAGE_IDS, _CACHED_FAISS_INDEX, _CACHED_VECTORS
    global _CACHED_PAGE_DENSE_PAGE_IDS, _CACHED_PAGE_DENSE_VECTORS
    global _CACHED_BM25_CHUNK, _CACHED_BM25_TITLE, _CACHED_BM25_PAGE
    global _CACHED_CROSS_ENCODER, _CACHED_PAGE_LOOKUP
    _CACHED_PAGE_IDS = None
    _CACHED_FAISS_INDEX = None
    _CACHED_VECTORS = None
    _CACHED_PAGE_DENSE_PAGE_IDS = None
    _CACHED_PAGE_DENSE_VECTORS = None
    _CACHED_BM25_CHUNK = None
    _CACHED_BM25_TITLE = None
    _CACHED_BM25_PAGE = None
    _CACHED_CROSS_ENCODER = None
    _CACHED_PAGE_LOOKUP = None


def _get_page_ids(root: Path) -> np.ndarray:
    global _CACHED_PAGE_IDS
    if _CACHED_PAGE_IDS is None:
        _CACHED_PAGE_IDS = np.load(root / "page_ids.npy")
    return _CACHED_PAGE_IDS


def _get_bm25(prefix: str, artifacts_dir: Optional[Path]):
    global _CACHED_BM25_CHUNK, _CACHED_BM25_TITLE, _CACHED_BM25_PAGE
    root = resolve_artifacts_dir(artifacts_dir)
    if prefix == "chunk":
        if _CACHED_BM25_CHUNK is None:
            _CACHED_BM25_CHUNK = load_bm25_index(root, prefix="chunk")
        return _CACHED_BM25_CHUNK
    if prefix == "title":
        if _CACHED_BM25_TITLE is None:
            _CACHED_BM25_TITLE = load_bm25_index(root, prefix="title")
        return _CACHED_BM25_TITLE
    if prefix == "page":
        if _CACHED_BM25_PAGE is None:
            _CACHED_BM25_PAGE = load_bm25_index(root, prefix="page")
        return _CACHED_BM25_PAGE
    raise ValueError(f"unknown bm25 prefix: {prefix}")


def _get_faiss_index(artifacts_dir: Optional[Path]):
    global _CACHED_FAISS_INDEX
    if _CACHED_FAISS_INDEX is None:
        _CACHED_FAISS_INDEX, _, _ = load_index(artifacts_dir)
    return _CACHED_FAISS_INDEX


def _get_cross_encoder(model_name: str):
    global _CACHED_CROSS_ENCODER
    if CrossEncoder is None:
        raise ImportError("sentence-transformers is required for cross-encoder reranking.")
    if _CACHED_CROSS_ENCODER is None:
        _CACHED_CROSS_ENCODER = CrossEncoder(model_name, device="cpu")
    return _CACHED_CROSS_ENCODER


def _get_page_lookup(root: Path) -> Dict[int, Tuple[str, str]]:
    global _CACHED_PAGE_LOOKUP
    if _CACHED_PAGE_LOOKUP is None:
        data = np.load(root / PAGE_FEATURES_NAME, allow_pickle=True)
        lookup: Dict[int, Tuple[str, str]] = {}
        for pid, title, content in zip(data["page_ids"], data["titles"], data["contents"]):
            lookup[int(pid)] = (str(title).strip(), str(content).lower())
        _CACHED_PAGE_LOOKUP = lookup
    return _CACHED_PAGE_LOOKUP


def _page_ranking_from_chunk_scores(
    chunk_indices: np.ndarray,
    chunk_scores: np.ndarray,
    page_ids: np.ndarray,
    *,
    agg: str = "max",
) -> List[int]:
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
    scores: Dict[int, float] = {}
    if weights is None:
        weights = [1.0] * len(rankings)
    for ranking, weight in zip(rankings, weights):
        for rank, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + weight / (rrf_k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in fused[:top_k]]


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
    # אופטימיזציה 2: העמקת החיפוש בגרף לקבלת מועמדים איכותיים יותר
    index.hnsw.efSearch = max(512, max(ef_min, min(candidate_k, ef_cap)))
    distances, indices = index.search(query_vectors, candidate_k)
    out: List[List[int]] = []
    for i in range(len(query_vectors)):
        out.append(_page_ranking_from_chunk_scores(indices[i], distances[i], page_ids, agg=agg))
    return out


def _bm25_chunk_page_ranking(bm25_index, query: str, *, candidate_k: int, agg: str) -> List[int]:
    doc_idx, doc_scores = bm25_index.search(query, top_k=candidate_k)
    return _page_ranking_from_chunk_scores(doc_idx, doc_scores, bm25_index.page_ids, agg=agg)


def _bm25_page_level_ranking(bm25_index, query: str, *, candidate_k: int) -> List[int]:
    doc_idx, _ = bm25_index.search(query, top_k=candidate_k)
    return [int(bm25_index.page_ids[int(i)]) for i in doc_idx]


def _merged_bm25_query(q_orig: str, q_kw: str) -> str:
    if q_orig == q_kw:
        return q_orig
    seen: Set[str] = set()
    parts: List[str] = []
    for tok in tokenize(q_orig) + tokenize(q_kw):
        if tok not in seen:
            seen.add(tok)
            parts.append(tok)
    return " ".join(parts) if parts else q_orig


def _bm25_expanded_page_ranking(
    bm25_index, q_orig: str, q_kw: str, *, candidate_k: int
) -> List[int]:
    query = _merged_bm25_query(q_orig, q_kw)
    return _bm25_page_level_ranking(bm25_index, query, candidate_k=candidate_k)


def _simple_search_batch(
    queries: List[str],
    *,
    top_k: int,
    artifacts_dir: Optional[Path],
) -> List[List[int]]:
    hp = load_hparams()
    root = resolve_artifacts_dir(artifacts_dir)

    page_ids = _get_page_ids(root)
    query_vectors = embed_queries(queries)

    candidate_k = max(top_k * max(1, int(hp_get(hp, "retrieve.candidate_multiplier", 50))), top_k)
    bm25_candidate_k = max(top_k * max(1, int(hp_get(hp, "retrieve.bm25_candidate_multiplier", 50))), top_k)
    
    # אופטימיזציה 3: משחק עדין בין 12 ל-14 (אפשר לנסות את שניהם)
    rerank_cap = 12 
    
    use_bm25 = bool(hp_get(hp, "retrieve.use_bm25", True))
    use_title = bool(hp_get(hp, "retrieve.use_title_bm25", True)) and has_bm25_index(root, "title")
    use_page = bool(hp_get(hp, "retrieve.use_page_bm25", True)) and has_bm25_index(root, "page")
    use_expansion = bool(hp_get(hp, "retrieve.use_query_expansion", True))
    
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_model = _get_cross_encoder(model_name)
    page_lookup = _get_page_lookup(root)
        
    rrf_k = int(hp_get(hp, "retrieve.rrf_k", 15))
    ef_min = int(hp_get(hp, "faiss_hnsw.ef_search_min", 128))
    ef_cap = int(hp_get(hp, "faiss_hnsw.ef_search_cap", 256))
    agg = str(hp_get(hp, "retrieve.page_aggregation", "max_plus_mean_top3"))

    w_dense = float(hp_get(hp, "retrieve.dense_rrf_weight", 1.0))
    w_chunk = float(hp_get(hp, "retrieve.bm25_chunk_rrf_weight", 1.2))
    w_title = float(hp_get(hp, "retrieve.title_bm25_rrf_weight", 1.8))
    w_page = float(hp_get(hp, "retrieve.page_bm25_rrf_weight", 1.0))

    dense_rankings = _dense_hnsw_rankings(
        query_vectors, page_ids, _get_faiss_index(root), candidate_k=candidate_k, ef_min=ef_min, ef_cap=ef_cap, agg=agg
    )

    bm25_chunk = _get_bm25("chunk", artifacts_dir) if use_bm25 and has_bm25_index(root, "chunk") else None
    bm25_title = _get_bm25("title", artifacts_dir) if use_bm25 and use_title else None
    bm25_page = _get_bm25("page", artifacts_dir) if use_bm25 and use_page else None

    query_pools: List[List[int]] = []
    orig_queries_processed: List[str] = []
    
    for i, query in enumerate(queries):
        q_orig, q_kw = query_versions(query, use_expansion=use_expansion)
        orig_queries_processed.append(q_orig)

        rankings: List[List[int]] = [dense_rankings[i]]
        weights: List[float] = [w_dense]

        if bm25_chunk is not None:
            rankings.append(_bm25_chunk_page_ranking(bm25_chunk, q_orig, candidate_k=bm25_candidate_k, agg=agg))
            weights.append(w_chunk)
        if bm25_title is not None:
            rankings.append(_bm25_expanded_page_ranking(bm25_title, q_orig, q_kw, candidate_k=bm25_candidate_k))
            weights.append(w_title)
        if bm25_page is not None:
            rankings.append(_bm25_expanded_page_ranking(bm25_page, q_orig, q_kw, candidate_k=bm25_candidate_k))
            weights.append(w_page)

        pool = _rrf_fuse(rankings, rrf_k=rrf_k, top_k=rerank_cap, weights=weights)
        query_pools.append(pool)

    all_pairs: List[Tuple[str, str]] = []
    pool_sizes: List[int] = []
    
    for q_orig, pool in zip(orig_queries_processed, query_pools):
        pool_sizes.append(len(pool))
        for pid in pool:
            title, content_lower = page_lookup.get(pid, ("", ""))
            words = content_lower.split()
            # חזרה מדויקת לפורמט ה-120 מילים הבטוח שהביא 0.2838
            snippet = " ".join(words[:120]) 
            summary = f"Title: {title}. Context: {snippet}" if title else snippet
            all_pairs.append((q_orig, summary))

    all_scores = cross_encoder_model.predict(all_pairs, batch_size=128, show_progress_bar=False) if all_pairs else []

    results: List[List[int]] = []
    score_idx = 0
    
    for pool, p_size in zip(query_pools, pool_sizes):
        if p_size == 0:
            results.append([])
            continue
        
        sub_scores = all_scores[score_idx : score_idx + p_size]
        score_idx += p_size
        
        ranked = sorted(zip(pool, sub_scores), key=lambda x: x[1], reverse=True)
        results.append([pid for pid, _ in ranked[:top_k]])

    return results


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    if not queries:
        return []
    return _simple_search_batch(queries, top_k=top_k, artifacts_dir=artifacts_dir)