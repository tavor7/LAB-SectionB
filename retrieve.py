
"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["TORCH_NUM_THREADS"] = "4" 

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
except ImportError:  # pragma: no cover
    CrossEncoder = None  # type: ignore

PAGE_FEATURES_NAME = "page_features.npz"

_CACHED_PAGE_IDS = None
_CACHED_FAISS_INDEX = None
_CACHED_BM25_CHUNK = None
_CACHED_BM25_TITLE = None
_CACHED_BM25_PAGE = None
_CACHED_CROSS_ENCODER = None
_CACHED_PAGE_LOOKUP: Optional[Dict[int, Tuple[str, str]]] = None


def clear_retrieve_cache() -> None:
    """Reset loaded indexes."""
    global _CACHED_PAGE_IDS, _CACHED_FAISS_INDEX
    global _CACHED_BM25_CHUNK, _CACHED_BM25_TITLE, _CACHED_BM25_PAGE
    global _CACHED_CROSS_ENCODER, _CACHED_PAGE_LOOKUP
    _CACHED_PAGE_IDS = None
    _CACHED_FAISS_INDEX = None
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
        if agg == "max_plus_mean_top3":
            page_scores[pid] = float(0.2 * scores[0] + 0.8 * np.mean(scores[:3]))
        else:
            page_scores[pid] = float(scores[0])

    items = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in items]


def _rrf_fuse_with_scores(
    rankings: List[List[int]],
    *,
    rrf_k: int,
    top_k: int,
    weights: Optional[List[float]] = None,
) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    if weights is None:
        weights = [1.0] * len(rankings)
    for ranking, weight in zip(rankings, weights):
        for rank, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + weight / (rrf_k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]


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


def _get_smart_snippet(content_lower: str, query_tokens: List[str], window_size: int = 120) -> str:
    words = content_lower.split()
    if len(words) <= window_size:
        return " ".join(words)
    
    q_set = set(query_tokens)
    best_start = 0
    max_matches = -1
    
    for start_idx in range(0, len(words) - window_size + 1, 20):
        window = words[start_idx : start_idx + window_size]
        matches = sum(1 for w in window if w in q_set)
        if matches > max_matches:
            max_matches = matches
            best_start = start_idx
            
    return " ".join(words[best_start : best_start + window_size])


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
    
    rerank_cap = int(
    hp_get(hp, "retrieve.rerank_candidate_cap", 12)
)
    
    use_bm25 = bool(hp_get(hp, "retrieve.use_bm25", True))
    use_title = bool(hp_get(hp, "retrieve.use_title_bm25", True)) and has_bm25_index(root, "title")
    use_page = bool(hp_get(hp, "retrieve.use_page_bm25", True)) and has_bm25_index(root, "page")
    use_expansion = bool(hp_get(hp, "retrieve.use_query_expansion", True))
    
    model_name = str(
    hp_get(
        hp,
        "retrieve.cross_encoder_model",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ))
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
    cross_encoder_rrf_weight = float(
    hp_get(hp, "retrieve.cross_encoder_rrf_weight", 3.0))

    dense_rankings = _dense_hnsw_rankings(
        query_vectors, page_ids, _get_faiss_index(root), candidate_k=candidate_k, ef_min=ef_min, ef_cap=ef_cap, agg=agg
    )

    bm25_chunk = _get_bm25("chunk", artifacts_dir) if use_bm25 and has_bm25_index(root, "chunk") else None
    bm25_title = _get_bm25("title", artifacts_dir) if use_bm25 and use_title else None
    bm25_page = _get_bm25("page", artifacts_dir) if use_bm25 and use_page else None

    query_pools_with_scores: List[List[Tuple[int, float]]] = []
    enhanced_queries: List[str] = []
    query_token_lists: List[List[str]] = []
    
    for i, query in enumerate(queries):
        q_orig, q_kw = query_versions(query, use_expansion=use_expansion)
        q_enhanced = _merged_bm25_query(q_orig, q_kw)
        enhanced_queries.append(q_enhanced)
        query_token_lists.append(tokenize(q_enhanced))

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

        pool_with_score = _rrf_fuse_with_scores(rankings, rrf_k=rrf_k, top_k=rerank_cap, weights=weights)
        query_pools_with_scores.append(pool_with_score)

    all_pairs: List[Tuple[str, str]] = []
    pool_sizes: List[int] = []
    
    for q_enhanced, pool_with_score, q_tokens in zip(enhanced_queries, query_pools_with_scores, query_token_lists):
        pool_sizes.append(len(pool_with_score))
        for pid, _ in pool_with_score:
            title, content_lower = page_lookup.get(pid, ("", ""))
            snippet = _get_smart_snippet(content_lower, q_tokens, window_size=120)
            summary = f"Title: {title}. Context: {snippet}" if title else snippet
            all_pairs.append((q_enhanced, summary))

    all_scores = cross_encoder_model.predict(all_pairs, batch_size=128, show_progress_bar=False) if all_pairs else []

    results: List[List[int]] = []
    score_idx = 0
    
    for pool_with_score, p_size in zip(query_pools_with_scores, pool_sizes):
        if p_size == 0:
            results.append([])
            continue
        
        sub_cross_scores = all_scores[score_idx : score_idx + p_size]
        score_idx += p_size
        
        final_ranked_pool = []
        for idx, (pid, rrf_score) in enumerate(pool_with_score):
            combined_score = (float(sub_cross_scores[idx]) + cross_encoder_rrf_weight * float(rrf_score))
            final_ranked_pool.append((pid, combined_score))
        
        ranked = sorted(final_ranked_pool, key=lambda x: x[1], reverse=True)
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