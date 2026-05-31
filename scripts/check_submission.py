"""Verify repo is ready for grading (fresh-clone style checks)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED_ARTIFACTS = [
    "faiss.index",
    "page_ids.npy",
    "index_vectors.npy",
    "meta.json",
    "bm25_vocab.json",
    "bm25_idf.npy",
    "bm25_indptr.npy",
    "bm25_indices.npy",
    "bm25_data.npy",
    "bm25_doc_lens.npy",
    "bm25_avgdl.json",
    "bm25_page_ids.npy",
]


def main() -> int:
    errors: list[str] = []
    artifacts = ROOT / "artifacts"
    queries = ROOT / "data" / "public_queries.json"

    if not queries.is_file():
        errors.append(f"Missing {queries} (needed for eval_public.py)")

    for name in REQUIRED_ARTIFACTS:
        if not (artifacts / name).is_file():
            errors.append(f"Missing artifact: artifacts/{name}")

    try:
        from main import run

        out = run(["test query"])
        if not isinstance(out, list) or not out or not isinstance(out[0], list):
            errors.append("main.run() did not return list[list[int]]")
    except Exception as exc:
        errors.append(f"main.run() failed: {exc}")

    if errors:
        print("check_submission: FAILED")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("check_submission: OK (artifacts present, run() works)")
    print("Next: python scripts/eval_public.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
