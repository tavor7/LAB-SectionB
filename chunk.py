"""Optional preprocessing and chunking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from config import hp_get, load_hparams
from utils import entry_text


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def _word_chunks(
    text: str,
    *,
    chunk_words: int,
    overlap_words: int,
) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    if chunk_words <= 0:
        return [" ".join(words)]
    overlap_words = max(0, min(overlap_words, max(0, chunk_words - 1)))
    step = max(1, chunk_words - overlap_words)

    out: List[str] = []
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_words]
        if not piece:
            break
        out.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
    return out


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """
    Split one corpus entry into retrieval units.

    Default: title-only chunk + fixed-size word chunks with overlap.
    """
    page_id = int(record["page_id"])
    title = str(record.get("title", "")).strip()
    content = str(record.get("content", "")).strip()

    hp = load_hparams()
    chunk_words = int(hp_get(hp, "chunking.chunk_words", 140))
    overlap_words = int(hp_get(hp, "chunking.overlap_words", 35))
    title_chunk_enabled = bool(hp_get(hp, "chunking.title_chunk", True))

    chunks: List[Chunk] = []
    if title_chunk_enabled and title:
        chunks.append(Chunk(page_id=page_id, chunk_id=-1, text=title))

    content_chunks = _word_chunks(
        content, chunk_words=chunk_words, overlap_words=overlap_words
    )
    for i, c in enumerate(content_chunks):
        if title:
            text = f"{title}\n\n{c}".strip()
        else:
            text = c.strip()
        chunks.append(Chunk(page_id=page_id, chunk_id=i, text=text))

    if not chunks:
        chunks = [Chunk(page_id=page_id, chunk_id=0, text=entry_text(record))]
    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
