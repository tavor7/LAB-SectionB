# Section B — Multi-index hybrid retrieval

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
| `retrieve.py` | Query-time multi-index retrieval + feature rerank |
| `query_expand.py` | Stopword-stripped keyword queries for BM25 |
| `embed.py` | MiniLM query/chunk embeddings |
| `index.py` | Offline FAISS + BM25 artifact writers (not timed at grading) |
| `lexical.py` | BM25 build/load/search (chunk, title, page indexes) |
| `chunk.py` | Title + body word-window chunking (`Title:/Content:` format) |
| `config.py` / `hparams.json` | Hyperparameters |
| `utils.py` | Paths, corpus helpers |
| `eval.py` | NDCG metrics (course file — do not edit) |
| `scripts/eval_public.py` | Self-test on 50 public queries |
| `scripts/build_index.py` | Offline full / incremental index build |
| `artifacts/` | **Required** precomputed index (see below) |
| `data/public_queries.json` | Public eval queries + labels (small; in repo) |

The full Wikipedia corpus (`data/Wikipedia Entries/`) is **not** in git; it is only needed to **rebuild** the index locally.

## Artifacts (required in repo)

`run()` loads from `artifacts/`:

| File | Purpose |
|------|------|
| `faiss.index` | FAISS HNSW (inner product on L2-normalized vectors) |
| `page_ids.npy` | int64: embedding row → `page_id` |
| `index_vectors.npy` | float32 chunk embeddings (brute dense mode) |
| `meta.json` | Model name, dims, chunking metadata |
| `bm25_chunk_*` | BM25 over content chunks (primary lexical index) |
| `bm25_title_*` | BM25 over title-only documents (one per page) |
| `bm25_page_*` | BM25 over full-page documents (one per page) |
| `page_features.npz` | `page_id`, `title`, `content` for rerank features |
| `bm25_vocab.json` … `bm25_page_ids.npy` | Legacy chunk BM25 names (alias of `bm25_chunk_*`) |

Large binaries are tracked with **Git LFS**. After clone, always run `git lfs pull`.

## Retrieval pipeline (query time)

Three-way **weighted RRF** fusion (no hand-crafted feature rerank on the public path):

1. **Dense chunks (HNSW)** — `all-MiniLM-L6-v2` on the original query; top chunk hits aggregated per page with `max_plus_mean_top3`.
2. **BM25 chunk** — same 140/35 word chunks as dense; **original query only** (matches legacy chunk BM25).
3. **BM25 page** — one doc per page (`Title: …\nContent: full text`); merged original + keyword query (stopwords removed).

Title-only BM25 is built offline but **disabled at query time** on the public eval — it hurt NDCG when fused. Page-level BM25 is the main gain over legacy (~0.236 → ~0.252 mean NDCG@10).

If `page_features.npz` or prefixed BM25 indexes are missing, `retrieve.py` **falls back** to the legacy pipeline: dense + single chunk BM25 + weighted RRF.

## Local setup (developers only)

```bash
pip install -r requirements.txt
```

Corpus: unzip handout into `data/Wikipedia Entries/` (same layout as the assignment).

### Build / extend index (offline, not timed)

```bash
python scripts/build_index.py
```

When dense artifacts and the build checkpoint are already complete, the script **skips re-embedding** and only builds `bm25_title_*`, `bm25_page_*`, `page_features.npz`, and `bm25_chunk_*` aliases (unless missing).

Long first-time runs: use `nohup` and checkpointing under `artifacts/shards/`. Resume by rerunning the same command if `hparams.json` chunking settings are unchanged.

### Evaluate on public queries

```bash
python scripts/eval_public.py
```

Prints `mean_ndcg@10` and `query_phase_time` (must stay under 60s for the full batch at grading).

Optional artifact root (for sweep variants):

```bash
ARTIFACTS_DIR=artifacts_sweep/w240_o60 python scripts/eval_public.py
# or
python scripts/eval_public.py --artifacts-dir artifacts_sweep/w240_o60
```

### Chunk-size sweep (local)

Builds persist under `artifacts_sweep/w{chunk}_o{overlap}/` with `manifest.json` (gitignored). Each variant is loadable for eval without overwriting others.

**Active sweep grid** (`build --all-grid`): **400/100**, **320/80**, **240/60** (Title/Content format).

**Legacy baseline** (not rebuilt): **140/35** in `artifacts/` — register with `sweep register`, still evaluable alongside sweep variants.

```bash
python scripts/sweep_chunk_sizes.py register    # register current artifacts/ as w140_o35
python scripts/sweep_chunk_sizes.py list
python scripts/sweep_chunk_sizes.py build --chunk-words 240   # one variant (long offline)
python scripts/sweep_chunk_sizes.py build --all-grid          # builds 400, 320, 240 only
python scripts/sweep_chunk_sizes.py register                # keep w140 (artifacts/) for eval
python scripts/sweep_chunk_sizes.py eval --folds 5            # compare all complete variants
```

Pick winner by **median fold NDCG**, then copy that directory into `artifacts/` for submission.

Full unattended pipeline (build all grid sizes → eval → ship):

```bash
nohup python -u scripts/sweep_chunk_sizes.py build --all-grid > build_sweep.log 2>&1 &
nohup bash scripts/watch_sweep_finish.sh > watch_sweep.log 2>&1 &
# or one shot after builds exist:
python scripts/sweep_chunk_sizes.py run-all --ship
```

### Dev tuning (small corpus)

```bash
BUILD_DEV_PUBLIC=1 BUILD_DEV_NUM_QUERIES=10 BUILD_DEV_NEG_PAGES=3000 python -u scripts/build_index.py
DEV_EVAL_NUM_QUERIES=10 python -u eval_dev.py
```

## Collaboration (pair grading)

See **[AUTHORS.md](AUTHORS.md)**. Git history must show **meaningful commits from both partners** — not a single dump at the deadline.

## Submit

Public GitHub repo: this code, `data/public_queries.json`, LFS-backed `artifacts/`, and this README. See the assignment PDF for the video and full grading rubric.
