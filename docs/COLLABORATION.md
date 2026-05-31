# Pair collaboration — git workflow

Section B repo grading includes **10% for evident pair work**: both members need **meaningful commits** in history, not one person pushing everything at the end.

## Before the deadline

1. Fill in **[AUTHORS.md](../AUTHORS.md)** with both names and GitHub usernames.
2. Agree who owns which areas (suggested split below).
3. Each person commits from **their own machine** (or VM) with **their own** `git config user.name` / `user.email`.

## Suggested commit split

| Partner | Suggested commits |
|---------|-------------------|
| **A** | Chunking (`chunk.py`), `config.py`, `hparams.json`, `eval_dev.py` |
| **B** | Index build (`index.py`), BM25 (`lexical.py`), retrieval (`retrieve.py`, `embed.py`) |
| **Both** | README, `scripts/eval_public.py`, artifact submission, review PRs on each other’s branches |

Each person should have **at least 3 commits** with real code or doc changes.

## Partner checklist (run on your laptop)

```bash
git clone https://github.com/tavor7/LAB-SectionB.git
cd LAB-SectionB
git lfs pull

git config user.name "Your Name"
git config user.email "you@example.com"

git checkout -b partner/<your-name>
# edit your files, then:
git add <files>
git commit -m "Brief message: what you changed and why"
git push origin partner/<your-name>
```

Open a pull request on GitHub; the other partner reviews and merges to `main`.

## Verify before submit

```bash
git shortlog -sn          # should list BOTH names with multiple commits each
git log --oneline --graph
python scripts/eval_public.py
```

## Co-authored commits (optional)

If you pair-program one change, use a trailer:

```bash
git commit -m "Improve RRF fusion weights

Co-authored-by: Name <email@example.com>"
```
