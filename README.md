# Section B — Multi-index hybrid retrieval + cross-encoder rerank

Wikipedia **page retrieval** for Section B. The autograder calls `main.run(queries)` once per batch; this repo ships **prebuilt `artifacts/`** (Git LFS) so staff do not rebuild at grading time.

**Current public benchmark** (`artifacts/`, paragraph index):

| Metric | Value |
|--------|-------|
| `mean_ndcg@10` | **0.4827** (29 public queries in `data/public_queries.json`) |
| `query_phase_time` | **~22s** (22.0s measured; limit 60s) |
| Index vectors | **211,665** paragraph chunks |

### What lives where

```
LAB-SectionB/
├── main.py, retrieve.py, …     # grading entry points
├── hparams.json, config.py     # hyperparameters
├── artifacts/                  # submission index (Git LFS)
├── data/public_queries.json    # public eval queries
├── scripts/
│   ├── eval_public.py          # public self-test (grading-style)
│   └── build_index.py          # offline index build (handout; not run at grading)
├── logs/                       # build logs (gitignored, local only)
├── local/                      # handout copies (gitignored, local only)
└── artifacts_backup*/          # old index backups (gitignored, local only)
```

---

## Quick start (matches grading)

Dependencies are assumed **already installed** (`numpy`, `sentence-transformers`, `faiss-cpu`, `torch`). No `pip install` during grading.

```bash
git clone https://github.com/tavor7/LAB-SectionB.git
cd LAB-SectionB
git lfs pull                    # required: large artifacts
python scripts/eval_public.py     # mean NDCG@10 + query time
```

---

## Solution overview

| Stage | What changed | Notes |
|-------|----------------|-------|
| 1. Baseline hybrid | Dense (MiniLM) + BM25 chunk, RRF | ~0.24 NDCG |
| 2. Multi-index | + BM25 title & page, tuned RRF | Stronger recall |
| 3. Cross-encoder | Batched CE rerank on RRF pool | Major NDCG gain |
| 4. Smart snippets | Query-aligned 120-word CE context | Better rerank input |
| 5. **Paragraph index** | Paragraph packing (max 400 words/chunk) + 100-word overlap | **0.4827 NDCG** (current `artifacts/`) |

**Current submission:** paragraph chunking (merge `\n\n` paragraphs up to **400 words max** per chunk, **100-word** overlap), multi-index RRF recall, cross-encoder rerank with smart snippets, checkpointed offline build.

---

## Query-time pipeline

```mermaid
flowchart TD
  Q[Batch of queries] --> Embed[MiniLM embed queries]
  Embed --> Dense[HNSW dense chunk search]
  Q --> BM25[BM25 chunk / title / page]
  BM25 --> Expand[Query expansion on title/page]
  Dense --> Agg["Page agg: max_plus_mean_top3"]
  Agg --> RRF[Weighted RRF fuse per query]
  Expand --> RRF
  RRF --> Pool["Top rerank_candidate_cap pages"]
  Pool --> Snip["Smart 120-word snippet"]
  Snip --> CE[Cross-encoder ms-marco-MiniLM-L-6-v2]
  CE --> Score["CE + cross_encoder_rrf_weight x RRF"]
  Score --> Top10[Top 10 page_ids]
```

**Per query:**

1. **Dense (HNSW)** — `all-MiniLM-L6-v2` on paragraph chunks; page aggregation `max_plus_mean_top3` (0.2 max + 0.8 mean top-3).
2. **BM25 chunk** — original query on paragraph chunks (`Title:/Content:` format).
3. **BM25 title** — page-level title index (query expansion).
4. **BM25 page** — full-page index (query expansion).
5. **Weighted RRF** — fuse four rankings (`rrf_k=20`; weights in `hparams.json`).
6. **Smart snippet** — 120-word window with most query-token overlap (step 20).
7. **Cross-encoder rerank** — `CE_score + cross_encoder_rrf_weight × rrf_score`; return top 10.

---

## Repository layout

| Path | Role |
|------|------|
| `main.py` | `run(queries)` → ranked `page_id` lists |
| `retrieve.py` | Multi-index RRF + cross-encoder rerank |
| `chunk.py` | Paragraph or word-window chunking |
| `embed.py` | MiniLM query/chunk embeddings |
| `index.py` | Offline FAISS + BM25 writers (checkpointed) |
| `lexical.py` | BM25 build/load/search |
| `query_expand.py` | Stopword-stripped keyword queries for BM25 |
| `config.py` / `hparams.json` | Hyperparameters |
| `eval.py` | NDCG metrics (course file) |
| `scripts/eval_public.py` | Public queries, NDCG@10 (grading-style self-test) |
| `scripts/build_index.py` | Offline full index build (handout; not run at grading) |
| `artifacts/` | **Submission index** (Git LFS) |
| `data/public_queries.json` | Public eval queries |

Corpus `data/Wikipedia Entries/` is **not** in git (handout only; needed to rebuild).

---

## Key hyperparameters (`hparams.json`)

**Chunking (current index):**

```json
"chunking": {
  "mode": "paragraph",
  "max_chunk_words": 400,
  "overlap_words": 100,
  "title_chunk": true
}
```

Must match `artifacts/meta.json` → `chunking.strategy`.

**Build-time (baked into index):**

| Group | Keys | Role |
|-------|------|------|
| `faiss_hnsw` | `M`, `ef_construction` | HNSW graph quality at build |
| `bm25` | `k1`, `b` | BM25 scoring in all lexical indexes |

**Query-time:**

| Key | Value | Notes |
|-----|-------|-------|
| `candidate_multiplier` | 400 | Dense/BM25 pool depth |
| `rerank_candidate_cap` | 20 | CE pool size per query |
| `cross_encoder_rrf_weight` | 3.0 | RRF blend into final CE score |
| `faiss_hnsw.ef_search_floor` | 512 | HNSW search depth floor |

---

## Offline build (not timed at grading)

```bash
pip install -r requirements.txt    # developers only
# unzip corpus to data/Wikipedia Entries/
python scripts/build_index.py
```

**Crash-safe / long runs:** checkpoints every 200 pages under `artifacts/shards/` + `build_checkpoint.json`. Re-run the same command to resume (invalidates if chunking changes).

**Backup before rebuild:**

```bash
mv artifacts artifacts_backup_word140_o35
mkdir artifacts
python scripts/build_index.py
```

---

## Video presentation

Project walkthrough (architecture, hyperparameters, engineering journey):

https://drive.google.com/file/d/1nvgg49NIbjJFGhySzKoMrGJePG-9vlj5/view?usp=sharing

