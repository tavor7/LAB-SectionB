"""BM25 lexical retrieval (stdlib + numpy only)."""
from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import ARTIFACTS_DIR

# Legacy (chunk index) filenames — grading / check_submission compatibility.
BM25_VOCAB_NAME = "bm25_vocab.json"
BM25_IDF_NAME = "bm25_idf.npy"
BM25_INDPTR_NAME = "bm25_indptr.npy"
BM25_INDICES_NAME = "bm25_indices.npy"
BM25_DATA_NAME = "bm25_data.npy"
BM25_DOC_LENS_NAME = "bm25_doc_lens.npy"
BM25_AVGDL_NAME = "bm25_avgdl.json"
BM25_PAGE_IDS_NAME = "bm25_page_ids.npy"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def log_build_progress(
    current: int,
    total: int,
    label: str,
    *,
    width: int = 36,
) -> None:
    """Text progress bar for offline build logs (nohup-friendly, stdlib only)."""
    if total <= 0:
        return
    current = min(max(0, current), total)
    pct = current / total
    filled = int(width * pct)
    bar = "=" * filled + "." * (width - filled)
    print(
        f"[build] {label} [{bar}] {current}/{total} ({100 * pct:.1f}%)",
        flush=True,
    )


def _progress_step(total: int, *, min_interval: int = 200) -> int:
    if total <= 0:
        return 1
    return max(1, min(min_interval, total // 40 or 1))


def _artifact_names(prefix: Optional[str]) -> Dict[str, str]:
    """Map logical artifact keys to filenames for a BM25 index."""
    if prefix is None:
        return {
            "vocab": BM25_VOCAB_NAME,
            "idf": BM25_IDF_NAME,
            "indptr": BM25_INDPTR_NAME,
            "indices": BM25_INDICES_NAME,
            "data": BM25_DATA_NAME,
            "doc_lens": BM25_DOC_LENS_NAME,
            "avgdl": BM25_AVGDL_NAME,
            "page_ids": BM25_PAGE_IDS_NAME,
        }
    p = prefix.strip()
    return {
        "vocab": f"bm25_{p}_vocab.json",
        "idf": f"bm25_{p}_idf.npy",
        "indptr": f"bm25_{p}_indptr.npy",
        "indices": f"bm25_{p}_indices.npy",
        "data": f"bm25_{p}_data.npy",
        "doc_lens": f"bm25_{p}_doc_lens.npy",
        "avgdl": f"bm25_{p}_avgdl.json",
        "page_ids": f"bm25_{p}_page_ids.npy",
    }


def has_bm25_index(artifacts_dir: Path, prefix: str) -> bool:
    """True if all files for the prefixed BM25 index exist."""
    names = _artifact_names(prefix)
    return all((artifacts_dir / names[k]).is_file() for k in names)


def _legacy_bm25_exists(artifacts_dir: Path) -> bool:
    return has_bm25_index(artifacts_dir, "legacy")


@dataclass
class BM25Index:
    vocab: Dict[str, int]
    idf: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    doc_lens: np.ndarray
    avgdl: float
    k1: float
    b: float
    page_ids: np.ndarray

    @property
    def n_docs(self) -> int:
        return int(len(self.doc_lens))

    @property
    def n_terms(self) -> int:
        return int(len(self.idf))

    def search(
        self,
        query: str,
        *,
        top_k: int = 2000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (doc_indices, scores) for top-k documents."""
        q_terms = tokenize(query)
        if not q_terms or self.n_docs == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        scores = np.zeros(self.n_docs, dtype=np.float32)
        for term in q_terms:
            tid = self.vocab.get(term)
            if tid is None:
                continue

            idf = float(self.idf[tid])
            start, end = int(self.indptr[tid]), int(self.indptr[tid + 1])
            if start >= end:
                continue

            doc_ids = self.indices[start:end]
            tfs = self.data[start:end]
            dls = self.doc_lens[doc_ids]
            denom = tfs + self.k1 * (1.0 - self.b + self.b * dls / self.avgdl)
            vals = (idf * (tfs * (self.k1 + 1.0)) / denom).astype(np.float32, copy=False)
            np.add.at(scores, doc_ids, vals)

        k = min(top_k, self.n_docs)
        if k <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        if k >= self.n_docs:
            order = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k - 1)[:k]
            order = part[np.argsort(-scores[part])]
        return order.astype(np.int64), scores[order]


def build_bm25_index(
    texts: Sequence[str],
    page_ids: np.ndarray,
    *,
    k1: float = 1.5,
    b: float = 0.75,
    progress_label: Optional[str] = None,
) -> BM25Index:
    n_docs = len(texts)
    if n_docs != len(page_ids):
        raise ValueError("texts and page_ids length mismatch")

    step = _progress_step(n_docs) if progress_label else 0
    if progress_label:
        log_build_progress(0, n_docs, f"{progress_label} tokenize")

    tokenized: List[List[str]] = []
    for doc_id, text in enumerate(texts):
        tokenized.append(tokenize(text))
        if progress_label and (
            doc_id % step == 0 or doc_id == n_docs - 1
        ):
            log_build_progress(
                doc_id + 1, n_docs, f"{progress_label} tokenize"
            )

    doc_lens = np.asarray([len(toks) for toks in tokenized], dtype=np.float32)
    avgdl = float(doc_lens.mean()) if n_docs else 0.0

    postings_lists: Dict[str, List[Tuple[int, float]]] = {}
    df: Dict[str, int] = {}

    if progress_label:
        log_build_progress(0, n_docs, f"{progress_label} postings")

    for doc_id, toks in enumerate(tokenized):
        tf_map = Counter(toks)
        for term, tf in tf_map.items():
            if term not in postings_lists:
                postings_lists[term] = []
                df[term] = 0
            postings_lists[term].append((doc_id, float(tf)))
            df[term] += 1
        if progress_label and (
            doc_id % step == 0 or doc_id == n_docs - 1
        ):
            log_build_progress(
                doc_id + 1, n_docs, f"{progress_label} postings"
            )

    vocab = {term: i for i, term in enumerate(sorted(postings_lists.keys()))}
    n_terms = len(vocab)

    idf = np.zeros(n_terms, dtype=np.float32)
    for term, tid in vocab.items():
        n = df[term]
        idf[tid] = math.log((n_docs - n + 0.5) / (n + 0.5) + 1.0)

    indptr = np.zeros(n_terms + 1, dtype=np.int64)
    indices_list: List[int] = []
    data_list: List[float] = []

    term_step = _progress_step(n_terms, min_interval=500) if progress_label else 0
    if progress_label:
        log_build_progress(0, n_terms, f"{progress_label} finalize")

    for tid in range(n_terms):
        term = next(t for t, i in vocab.items() if i == tid)
        plist = postings_lists[term]
        plist.sort(key=lambda x: x[0])
        for doc_id, tf in plist:
            indices_list.append(doc_id)
            data_list.append(tf)
        indptr[tid + 1] = len(indices_list)
        if progress_label and (
            tid % term_step == 0 or tid == n_terms - 1
        ):
            log_build_progress(
                tid + 1, n_terms, f"{progress_label} finalize"
            )

    if progress_label:
        log_build_progress(n_docs, n_docs, f"{progress_label} done")

    return BM25Index(
        vocab=vocab,
        idf=idf,
        indptr=indptr,
        indices=np.asarray(indices_list, dtype=np.int64),
        data=np.asarray(data_list, dtype=np.float32),
        doc_lens=doc_lens,
        avgdl=avgdl,
        k1=k1,
        b=b,
        page_ids=np.asarray(page_ids, dtype=np.int64),
    )


def save_bm25_index(
    index: BM25Index,
    artifacts_dir: Path,
    *,
    prefix: Optional[str] = None,
) -> None:
    """Save BM25 index. prefix=None uses legacy filenames; else bm25_{prefix}_*."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    names = _artifact_names(prefix)
    (artifacts_dir / names["vocab"]).write_text(
        json.dumps(index.vocab), encoding="utf-8"
    )
    np.save(artifacts_dir / names["idf"], index.idf)
    np.save(artifacts_dir / names["indptr"], index.indptr)
    np.save(artifacts_dir / names["indices"], index.indices)
    np.save(artifacts_dir / names["data"], index.data)
    np.save(artifacts_dir / names["doc_lens"], index.doc_lens)
    (artifacts_dir / names["avgdl"]).write_text(
        json.dumps({"avgdl": index.avgdl, "k1": index.k1, "b": index.b}),
        encoding="utf-8",
    )
    np.save(artifacts_dir / names["page_ids"], index.page_ids)


def copy_bm25_artifacts(
    artifacts_dir: Path,
    *,
    src_prefix: Optional[str],
    dst_prefix: Optional[str],
) -> None:
    """Copy BM25 files from one naming scheme to another (e.g. legacy → chunk)."""
    src = _artifact_names(src_prefix)
    dst = _artifact_names(dst_prefix)
    for key in src:
        shutil.copy2(artifacts_dir / src[key], artifacts_dir / dst[key])


def load_bm25_index(
    artifacts_dir: Optional[Path] = None,
    *,
    prefix: Optional[str] = None,
) -> BM25Index:
    """
    Load BM25 index from artifacts.

    prefix=None: try chunk index, then legacy bm25_* (grading layout).
    prefix='chunk'|'title'|'page': load that prefixed index.
    """
    root = artifacts_dir or ARTIFACTS_DIR

    resolved: Optional[str] = prefix
    if resolved is None:
        if has_bm25_index(root, "chunk"):
            resolved = "chunk"
        elif _legacy_bm25_exists(root):
            resolved = None
        else:
            raise FileNotFoundError(f"No BM25 index found under {root}")

    names = _artifact_names(None if resolved is None else resolved)

    vocab = json.loads((root / names["vocab"]).read_text(encoding="utf-8"))
    meta = json.loads((root / names["avgdl"]).read_text(encoding="utf-8"))
    return BM25Index(
        vocab=vocab,
        idf=np.load(root / names["idf"]),
        indptr=np.load(root / names["indptr"]),
        indices=np.load(root / names["indices"]),
        data=np.load(root / names["data"]),
        doc_lens=np.load(root / names["doc_lens"]),
        avgdl=float(meta["avgdl"]),
        k1=float(meta.get("k1", 1.5)),
        b=float(meta.get("b", 0.75)),
        page_ids=np.load(root / names["page_ids"]),
    )
