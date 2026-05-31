"""BM25 lexical retrieval (stdlib + numpy only)."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import ARTIFACTS_DIR

BM25_VOCAB_NAME = "bm25_vocab.json"
BM25_IDF_NAME = "bm25_idf.npy"
BM25_INDPTR_NAME = "bm25_indptr.npy"
BM25_INDICES_NAME = "bm25_indices.npy"
BM25_DATA_NAME = "bm25_data.npy"
BM25_DOC_LENS_NAME = "bm25_doc_lens.npy"
BM25_AVGDL_NAME = "bm25_avgdl.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


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
        """Return (doc_indices, scores) for top-k chunk documents."""
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
) -> BM25Index:
    n_docs = len(texts)
    if n_docs != len(page_ids):
        raise ValueError("texts and page_ids length mismatch")

    tokenized: List[List[str]] = [tokenize(t) for t in texts]
    doc_lens = np.asarray([len(toks) for toks in tokenized], dtype=np.float32)
    avgdl = float(doc_lens.mean()) if n_docs else 0.0

    # term -> list of (doc_id, tf)
    postings_lists: Dict[str, List[Tuple[int, float]]] = {}
    df: Dict[str, int] = {}

    for doc_id, toks in enumerate(tokenized):
        tf_map = Counter(toks)
        for term, tf in tf_map.items():
            if term not in postings_lists:
                postings_lists[term] = []
                df[term] = 0
            postings_lists[term].append((doc_id, float(tf)))
            df[term] += 1

    vocab = {term: i for i, term in enumerate(sorted(postings_lists.keys()))}
    n_terms = len(vocab)

    idf = np.zeros(n_terms, dtype=np.float32)
    for term, tid in vocab.items():
        n = df[term]
        idf[tid] = math.log((n_docs - n + 0.5) / (n + 0.5) + 1.0)

    indptr = np.zeros(n_terms + 1, dtype=np.int64)
    indices_list: List[int] = []
    data_list: List[float] = []

    for tid in range(n_terms):
        term = next(t for t, i in vocab.items() if i == tid)
        plist = postings_lists[term]
        plist.sort(key=lambda x: x[0])
        for doc_id, tf in plist:
            indices_list.append(doc_id)
            data_list.append(tf)
        indptr[tid + 1] = len(indices_list)

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


def save_bm25_index(index: BM25Index, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / BM25_VOCAB_NAME).write_text(
        json.dumps(index.vocab), encoding="utf-8"
    )
    np.save(artifacts_dir / BM25_IDF_NAME, index.idf)
    np.save(artifacts_dir / BM25_INDPTR_NAME, index.indptr)
    np.save(artifacts_dir / BM25_INDICES_NAME, index.indices)
    np.save(artifacts_dir / BM25_DATA_NAME, index.data)
    np.save(artifacts_dir / BM25_DOC_LENS_NAME, index.doc_lens)
    (artifacts_dir / BM25_AVGDL_NAME).write_text(
        json.dumps({"avgdl": index.avgdl, "k1": index.k1, "b": index.b}),
        encoding="utf-8",
    )
    np.save(artifacts_dir / "bm25_page_ids.npy", index.page_ids)


def load_bm25_index(artifacts_dir: Optional[Path] = None) -> BM25Index:
    root = artifacts_dir or ARTIFACTS_DIR
    vocab = json.loads((root / BM25_VOCAB_NAME).read_text(encoding="utf-8"))
    meta = json.loads((root / BM25_AVGDL_NAME).read_text(encoding="utf-8"))
    return BM25Index(
        vocab=vocab,
        idf=np.load(root / BM25_IDF_NAME),
        indptr=np.load(root / BM25_INDPTR_NAME),
        indices=np.load(root / BM25_INDICES_NAME),
        data=np.load(root / BM25_DATA_NAME),
        doc_lens=np.load(root / BM25_DOC_LENS_NAME),
        avgdl=float(meta["avgdl"]),
        k1=float(meta.get("k1", 1.5)),
        b=float(meta.get("b", 0.75)),
        page_ids=np.load(root / "bm25_page_ids.npy"),
    )
