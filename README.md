# Section B — Hybrid retrieval (dense + BM25)

Wikipedia page retrieval for Section B. The autograder calls `main.run(queries)`; this repo ships **prebuilt `artifacts/`** so staff do not rebuild the index at grading time.

## Quick start (matches grading)

Dependencies are assumed **already installed** (numpy, sentence-transformers, faiss-cpu). No `pip install` during grading.

```bash
git clone https://github.com/tavor7/LAB-SectionB.git
cd LAB-SectionB
git lfs pull                    # required: large artifacts use Git LFS
python scripts/eval_public.py     # must succeed without rebuilding the index
```

Expected output includes `mean_ndcg@10=...` and `query_phase_time=...`.

## Repository layout

| Path | Role |
|------|------|
| `main.py` | Entry point: `run(queries)` → ranked `page_id` lists |
| `retrieve.py` | Query-time hybrid search (dense + BM25 + RRF) |
| `embed.py` | MiniLM query/chunk embeddings |
| `index.py` | Offline FAISS + artifact writers (not timed at grading) |
| `lexical.py` | BM25 index build/load/search |
| `chunk.py` | Title + body word-window chunking |
| `config.py` / `hparams.json` | Hyperparameters |
| `utils.py` | Paths, corpus helpers |
| `eval.py` | NDCG metrics (course file — do not edit) |
| `scripts/eval_public.py` | Self-test on 50 public queries |
| `scripts/build_index.py` | Offline full index build |
| `artifacts/` | **Required** precomputed index (see below) |
| `data/public_queries.json` | Public eval queries + labels (small; in repo) |

The full Wikipedia corpus (`data/Wikipedia Entries/`) is **not** in git; it ships with the course handout and is only needed to **rebuild** the index locally.

## Artifacts (required in repo)

`run()` loads from `artifacts/`:

| File | Purpose |
|------|------|
| `faiss.index` | FAISS HNSW (inner product on L2-normalized vectors) |
| `page_ids.npy` | int64: embedding row → `page_id` |
| `index_vectors.npy` | float32 chunk embeddings (brute dense mode) |
| `meta.json` | Model name, dims, chunking metadata |
| `bm25_vocab.json` | Term → term id |
| `bm25_idf.npy`, `bm25_indptr.npy`, `bm25_indices.npy`, `bm25_data.npy` | BM25 inverted index (CSR) |
| `bm25_doc_lens.npy`, `bm25_avgdl.json` | BM25 length stats |
| `bm25_page_ids.npy` | `page_id` per BM25 document row |

Large binaries are tracked with **Git LFS**. After clone, always run `git lfs pull`.

## Retrieval design

1. **Chunking** — title-only chunk + overlapping body windows (`chunk.py`, `hparams.json`).
2. **Dense** — `sentence-transformers/all-MiniLM-L6-v2`; page score = aggregate of chunk scores (`retrieve.page_aggregation`).
3. **Lexical** — BM25 over the same chunks (`lexical.py`).
4. **Fusion** — weighted reciprocal rank fusion at page level (`retrieve.rrf_k`, `dense_rrf_weight`, `bm25_rrf_weight`).

Tune `hparams.json` for chunk sizes, candidate pool sizes, HNSW `ef_search`, and RRF weights.

## Local setup (developers only)

```bash
pip install -r requirements.txt
```

Corpus: unzip handout into `data/Wikipedia Entries/` (same layout as the assignment).

### Rebuild index (offline, not timed)

```bash
python scripts/build_index.py
```

Long runs: use `nohup` and checkpointing under `artifacts/shards/` (see assignment notes). Resume by rerunning the same command if `hparams.json` chunking settings are unchanged.

### Dev tuning (small corpus)

```bash
BUILD_DEV_PUBLIC=1 BUILD_DEV_NUM_QUERIES=10 BUILD_DEV_NEG_PAGES=3000 python -u scripts/build_index.py
DEV_EVAL_NUM_QUERIES=10 python -u eval_dev.py
```

## Collaboration (pair grading)

See **[AUTHORS.md](AUTHORS.md)**. Git history must show **meaningful commits from both partners** — not a single dump at the deadline.

## Submit

Public GitHub repo: this code, `data/public_queries.json`, LFS-backed `artifacts/`, and this README. See the assignment PDF for the video and full grading rubric.
