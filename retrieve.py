"""Query-time hybrid retrieval (multi-index + feature rerank)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import hp_get, load_hparams
from embed import embed_queries
from index import load_index
from lexical import has_bm25_index, load_bm25_index, tokenize
from query_expand import query_versions
from utils import ARTIFACTS_DIR, K_EVAL

PAGE_FEATURES_NAME = "page_features.npz"

_CACHED_PAGE_IDS = None
_CACHED_FAISS_INDEX = None
_CACHED_VECTORS = None
_CACHED_BM25_CHUNK = None
_CACHED_BM25_TITLE = None
_CACHED_BM25_PAGE = None
_CACHED_PAGE_LOOKUP: Optional[Dict[int, Tuple[str, str, frozenset]]] = None


def _get_page_ids(root: Path) -> np.ndarray:
    global _CACHED_PAGE_IDS
    if _CACHED_PAGE_IDS is None:
        _CACHED_PAGE_IDS = np.load(root / "page_ids.npy")
    return _CACHED_PAGE_IDS


def _get_bm25(prefix: str, artifacts_dir: Optional[Path]):
    global _CACHED_BM25_CHUNK, _CACHED_BM25_TITLE, _CACHED_BM25_PAGE
    root = artifacts_dir or ARTIFACTS_DIR
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


def _get_vectors(root: Path) -> np.ndarray:
    global _CACHED_VECTORS
    if _CACHED_VECTORS is None:
        _CACHED_VECTORS = np.load(root / "index_vectors.npy", mmap_mode="r")
    return _CACHED_VECTORS


def _get_page_lookup(root: Path) -> Dict[int, Tuple[str, str, frozenset]]:
    """page_id -> (title, content_lower, title_tokens)."""
    global _CACHED_PAGE_LOOKUP
    if _CACHED_PAGE_LOOKUP is None:
        data = np.load(root / PAGE_FEATURES_NAME, allow_pickle=True)
        lookup: Dict[int, Tuple[str, str, frozenset]] = {}
        for pid, title, content in zip(
            data["page_ids"], data["titles"], data["contents"]
        ):
            title_s = str(title)
            lookup[int(pid)] = (
                title_s,
                str(content).lower(),
                frozenset(tokenize(title_s)),
            )
        _CACHED_PAGE_LOOKUP = lookup
    return _CACHED_PAGE_LOOKUP


def _enhanced_artifacts_ready(root: Path, hp: dict) -> bool:
    if not (root / PAGE_FEATURES_NAME).is_file():
        return False
    if not has_bm25_index(root, "chunk") and not (root / "bm25_vocab.json").is_file():
        return False
    if bool(hp_get(hp, "retrieve.use_title_bm25", True)) and not has_bm25_index(
        root, "title"
    ):
        return False
    if bool(hp_get(hp, "retrieve.use_page_bm25", True)) and not has_bm25_index(
        root, "page"
    ):
        return False
    return True


def _page_ranking_from_chunk_scores(
    chunk_indices: np.ndarray,
    chunk_scores: np.ndarray,
    page_ids: np.ndarray,
    *,
    agg: str = "max",
) -> List[int]:
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


def _rrf_scores_from_ranking(
    ranking: List[int],
    *,
    rrf_k: int,
    weight: float = 1.0,
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for rank, pid in enumerate(ranking):
        scores[pid] = scores.get(pid, 0.0) + weight / (rrf_k + rank + 1)
    return scores


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


def _dense_brute_rankings(
    query_vectors: np.ndarray,
    page_ids: np.ndarray,
    vectors: np.ndarray,
    *,
    candidate_k: int,
    agg: str,
) -> List[List[int]]:
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


def _bm25_chunk_page_ranking(
    bm25_index,
    query: str,
    *,
    candidate_k: int,
    agg: str,
) -> List[int]:
    doc_idx, doc_scores = bm25_index.search(query, top_k=candidate_k)
    return _page_ranking_from_chunk_scores(
        doc_idx,
        doc_scores,
        bm25_index.page_ids,
        agg=agg,
    )


def _bm25_page_level_ranking(
    bm25_index,
    query: str,
    *,
    candidate_k: int,
) -> List[int]:
    doc_idx, _ = bm25_index.search(query, top_k=candidate_k)
    return [int(bm25_index.page_ids[int(i)]) for i in doc_idx]


def _merged_bm25_query(q_orig: str, q_kw: str) -> str:
    """Single BM25 query with deduped tokens from original + keyword versions."""
    if q_orig == q_kw:
        return q_orig
    seen: Set[str] = set()
    parts: List[str] = []
    for tok in tokenize(q_orig) + tokenize(q_kw):
        if tok not in seen:
            seen.add(tok)
            parts.append(tok)
    return " ".join(parts) if parts else q_orig


def _bm25_expanded_chunk_ranking(
    bm25_index,
    q_orig: str,
    q_kw: str,
    *,
    candidate_k: int,
    agg: str,
    use_dual_rrf: bool,
    rrf_k: int,
    pool_k: int,
) -> List[int]:
    if use_dual_rrf and q_orig != q_kw:
        return _bm25_dual_fused_chunk_ranking(
            bm25_index,
            q_orig,
            q_kw,
            candidate_k=candidate_k,
            agg=agg,
            rrf_k=rrf_k,
            pool_k=pool_k,
        )
    query = _merged_bm25_query(q_orig, q_kw)
    return _bm25_chunk_page_ranking(
        bm25_index, query, candidate_k=candidate_k, agg=agg
    )


def _bm25_expanded_page_ranking(
    bm25_index,
    q_orig: str,
    q_kw: str,
    *,
    candidate_k: int,
    use_dual_rrf: bool,
    rrf_k: int,
    pool_k: int,
) -> List[int]:
    if use_dual_rrf and q_orig != q_kw:
        return _bm25_dual_fused_page_ranking(
            bm25_index,
            q_orig,
            q_kw,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            pool_k=pool_k,
        )
    query = _merged_bm25_query(q_orig, q_kw)
    return _bm25_page_level_ranking(bm25_index, query, candidate_k=candidate_k)


def _bm25_dual_fused_chunk_ranking(
    bm25_index,
    q_orig: str,
    q_kw: str,
    *,
    candidate_k: int,
    agg: str,
    rrf_k: int,
    pool_k: int,
) -> List[int]:
    r1 = _bm25_chunk_page_ranking(
        bm25_index, q_orig, candidate_k=candidate_k, agg=agg
    )
    if q_orig == q_kw:
        return r1[:pool_k]
    r2 = _bm25_chunk_page_ranking(
        bm25_index, q_kw, candidate_k=candidate_k, agg=agg
    )
    return _rrf_fuse([r1, r2], rrf_k=rrf_k, top_k=pool_k, weights=[1.0, 1.0])


def _bm25_dual_fused_page_ranking(
    bm25_index,
    q_orig: str,
    q_kw: str,
    *,
    candidate_k: int,
    rrf_k: int,
    pool_k: int,
) -> List[int]:
    r1 = _bm25_page_level_ranking(bm25_index, q_orig, candidate_k=candidate_k)
    if q_orig == q_kw:
        return r1[:pool_k]
    r2 = _bm25_page_level_ranking(bm25_index, q_kw, candidate_k=candidate_k)
    return _rrf_fuse([r1, r2], rrf_k=rrf_k, top_k=pool_k, weights=[1.0, 1.0])


def _combined_rrf_scores(
    rankings: List[List[int]],
    weights: List[float],
    *,
    rrf_k: int,
    score_cap: int,
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, pid in enumerate(ranking[:score_cap]):
            scores[pid] = scores.get(pid, 0.0) + weight / (rrf_k + rank + 1)
    return scores


def _light_feature_rerank(
    candidates: List[Tuple[int, float]],
    *,
    page_lookup: Dict[int, Tuple[str, str, frozenset]],
    q_orig: str,
    q_tokens: Set[str],
    w_tov: float,
    w_tcov: float,
    w_phrase: float,
    top_k: int,
) -> List[int]:
    q_lower = q_orig.lower().strip()
    final: Dict[int, float] = {}

    for pid, base in candidates:
        title, content_lower, title_tokens = page_lookup.get(
            pid, ("", "", frozenset())
        )
        bonus = (
            w_tov * _title_overlap(q_tokens, title_tokens)
            + w_tcov * _token_coverage(q_tokens, title_tokens)
        )
        if w_phrase > 0.0 and len(q_lower) >= 3:
            if q_lower in title.lower():
                bonus += w_phrase
            if q_lower in content_lower:
                bonus += w_phrase
        final[pid] = base + bonus

    ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in ranked[:top_k]]


def _top_slice(ranking: List[int], n: int) -> List[int]:
    return ranking[:n] if len(ranking) > n else ranking


def _query_token_set(query: str) -> Set[str]:
    return set(tokenize(query))


def _title_overlap(query_tokens: Set[str], title_tokens: Set[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & title_tokens) / len(query_tokens)


def _token_coverage(query_tokens: Set[str], doc_tokens: Set[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


def _exact_phrase_bonus(
    query: str,
    title: str,
    content_lower: str,
) -> float:
    q = query.lower().strip()
    if len(q) < 3:
        return 0.0
    bonus = 0.0
    if q in title.lower():
        bonus += 1.0
    if q in content_lower:
        bonus += 1.0
    return bonus


def _legacy_search_batch(
    queries: List[str],
    *,
    top_k: int,
    artifacts_dir: Optional[Path],
) -> List[List[int]]:
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
    bm25_rrf_weight = float(
        hp_get(hp, "retrieve.bm25_rrf_weight", hp_get(hp, "retrieve.bm25_chunk_rrf_weight", 1.0))
    )

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

    bm25_index = load_bm25_index(root)
    ranked: List[List[int]] = []
    for i, query in enumerate(queries):
        doc_idx, doc_scores = bm25_index.search(query, top_k=bm25_candidate_k)
        bm25_ranking = _page_ranking_from_chunk_scores(
            doc_idx,
            doc_scores,
            bm25_index.page_ids,
            agg=agg,
        )
        fused = _rrf_fuse(
            [dense_rankings[i], bm25_ranking],
            rrf_k=rrf_k,
            top_k=top_k,
            weights=[dense_rrf_weight, bm25_rrf_weight],
        )
        ranked.append(fused)

    return ranked


def _enhanced_search_batch(
    queries: List[str],
    *,
    top_k: int,
    artifacts_dir: Optional[Path],
) -> List[List[int]]:
    hp = load_hparams()
    root = artifacts_dir or ARTIFACTS_DIR
    page_ids = _get_page_ids(root)
    query_vectors = embed_queries(queries)

    mult = int(hp_get(hp, "retrieve.candidate_multiplier", 300))
    candidate_k = max(top_k * max(1, mult), top_k)
    bm25_mult = int(hp_get(hp, "retrieve.bm25_candidate_multiplier", 300))
    bm25_candidate_k = max(top_k * max(1, bm25_mult), top_k)
    rerank_cap = int(hp_get(hp, "retrieve.rerank_candidate_cap", 120))
    score_cap = int(hp_get(hp, "retrieve.rrf_score_cap", 500))
    rerank_cap = max(top_k, rerank_cap)
    score_cap = max(rerank_cap, score_cap)

    mode = str(hp_get(hp, "retrieve.mode", "hnsw")).lower()
    use_bm25 = bool(hp_get(hp, "retrieve.use_bm25", True))
    use_title = bool(hp_get(hp, "retrieve.use_title_bm25", True)) and has_bm25_index(
        root, "title"
    )
    use_page = bool(hp_get(hp, "retrieve.use_page_bm25", True)) and has_bm25_index(
        root, "page"
    )
    use_expansion = bool(hp_get(hp, "retrieve.use_query_expansion", True))
    rrf_k = int(hp_get(hp, "retrieve.rrf_k", 15))
    ef_min = int(hp_get(hp, "faiss_hnsw.ef_search_min", 128))
    ef_cap = int(hp_get(hp, "faiss_hnsw.ef_search_cap", 256))
    agg = str(hp_get(hp, "retrieve.page_aggregation", "max_plus_mean_top3"))

    w_dense = float(hp_get(hp, "retrieve.dense_rrf_weight", 1.0))
    w_chunk = float(hp_get(hp, "retrieve.bm25_chunk_rrf_weight", 1.2))
    w_title = float(hp_get(hp, "retrieve.title_bm25_rrf_weight", 1.8))
    w_page = float(hp_get(hp, "retrieve.page_bm25_rrf_weight", 1.0))
    w_tov = float(hp_get(hp, "retrieve.title_overlap_weight", 0.15))
    w_tcov = float(hp_get(hp, "retrieve.title_coverage_weight", 0.10))
    w_phrase = float(hp_get(hp, "retrieve.phrase_bonus_weight", 0.12))
    use_features = (w_tov + w_tcov + w_phrase) > 0.0
    page_lookup = _get_page_lookup(root) if use_features else None

    if mode == "brute":
        vec_path = root / "index_vectors.npy"
        if not vec_path.exists():
            raise FileNotFoundError(f"Brute mode requires {vec_path}.")
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

    bm25_chunk = _get_bm25("chunk", artifacts_dir) if use_bm25 else None
    bm25_title = _get_bm25("title", artifacts_dir) if use_title else None
    bm25_page = _get_bm25("page", artifacts_dir) if use_page else None

    use_dual_rrf = bool(hp_get(hp, "retrieve.use_dual_query_rrf", False))
    expand_chunk = bool(hp_get(hp, "retrieve.expand_chunk_bm25", False))

    results: List[List[int]] = []

    for i, query in enumerate(queries):
        q_orig, q_kw = query_versions(query, use_expansion=use_expansion)

        rankings: List[List[int]] = [dense_rankings[i]]
        weights: List[float] = [w_dense]

        if bm25_chunk is not None:
            if expand_chunk and use_expansion:
                rankings.append(
                    _bm25_expanded_chunk_ranking(
                        bm25_chunk,
                        q_orig,
                        q_kw,
                        candidate_k=bm25_candidate_k,
                        agg=agg,
                        use_dual_rrf=use_dual_rrf,
                        rrf_k=rrf_k,
                        pool_k=score_cap,
                    )
                )
            else:
                rankings.append(
                    _bm25_chunk_page_ranking(
                        bm25_chunk,
                        q_orig,
                        candidate_k=bm25_candidate_k,
                        agg=agg,
                    )
                )
            weights.append(w_chunk)
        if bm25_title is not None:
            if use_expansion:
                rankings.append(
                    _bm25_expanded_page_ranking(
                        bm25_title,
                        q_orig,
                        q_kw,
                        candidate_k=bm25_candidate_k,
                        use_dual_rrf=use_dual_rrf,
                        rrf_k=rrf_k,
                        pool_k=score_cap,
                    )
                )
            else:
                rankings.append(
                    _bm25_page_level_ranking(
                        bm25_title, q_orig, candidate_k=bm25_candidate_k
                    )
                )
            weights.append(w_title)
        if bm25_page is not None:
            if use_expansion:
                rankings.append(
                    _bm25_expanded_page_ranking(
                        bm25_page,
                        q_orig,
                        q_kw,
                        candidate_k=bm25_candidate_k,
                        use_dual_rrf=use_dual_rrf,
                        rrf_k=rrf_k,
                        pool_k=score_cap,
                    )
                )
            else:
                rankings.append(
                    _bm25_page_level_ranking(
                        bm25_page, q_orig, candidate_k=bm25_candidate_k
                    )
                )
            weights.append(w_page)

        if use_features and page_lookup is not None:
            pool = _rrf_fuse(
                rankings, rrf_k=rrf_k, top_k=rerank_cap, weights=weights
            )
            combined = _combined_rrf_scores(
                rankings, weights, rrf_k=rrf_k, score_cap=score_cap
            )
            top_candidates = [(pid, combined[pid]) for pid in pool if pid in combined]
            results.append(
                _light_feature_rerank(
                    top_candidates,
                    page_lookup=page_lookup,
                    q_orig=q_orig,
                    q_tokens=_query_token_set(q_orig),
                    w_tov=w_tov,
                    w_tcov=w_tcov,
                    w_phrase=w_phrase,
                    top_k=top_k,
                )
            )
        else:
            results.append(
                _rrf_fuse(
                    rankings,
                    rrf_k=rrf_k,
                    top_k=top_k,
                    weights=weights,
                )
            )

    return results


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """
    Rank pages for each query.

    Uses multi-index retrieval + feature rerank when extended artifacts exist;
    otherwise falls back to dense + single BM25 RRF.
    """
    if not queries:
        return []

    hp = load_hparams()
    root = artifacts_dir or ARTIFACTS_DIR

    if _enhanced_artifacts_ready(root, hp):
        return _enhanced_search_batch(
            queries, top_k=top_k, artifacts_dir=artifacts_dir
        )
    return _legacy_search_batch(
        queries, top_k=top_k, artifacts_dir=artifacts_dir
    )
