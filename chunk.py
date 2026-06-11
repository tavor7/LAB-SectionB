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


def format_chunk_text(title: str, content: str) -> str:
    """Shared dense/BM25 chunk wrapper (matches page-level BM25 in index.py)."""
    return f"Title: {title}\nContent: {content}"


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


def _apply_word_overlap(chunks: List[str], overlap_words: int) -> List[str]:
    """Prepend trailing words from the previous chunk onto the next chunk."""
    if overlap_words <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_words = chunks[i - 1].split()
        if len(prev_words) >= overlap_words:
            prefix = " ".join(prev_words[-overlap_words:])
        else:
            prefix = " ".join(prev_words)
        combined = f"{prefix} {chunks[i]}".strip() if prefix else chunks[i]
        out.append(combined)
    return out


def _paragraph_chunks(
    content: str,
    *,
    max_chunk_words: int,
    overlap_words: int,
) -> List[str]:
    """Pack paragraphs, then apply word-level overlap between consecutive chunks."""
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        return [""]

    packed: List[str] = []
    current: List[str] = []
    current_words = 0

    for para in paragraphs:
        pw = len(para.split())
        if pw > max_chunk_words:
            if current:
                packed.append("\n\n".join(current))
                current = []
                current_words = 0
            packed.extend(
                _word_chunks(
                    para,
                    chunk_words=max_chunk_words,
                    overlap_words=overlap_words,
                )
            )
            continue
        if current_words + pw > max_chunk_words and current:
            packed.append("\n\n".join(current))
            current = [para]
            current_words = pw
        else:
            current.append(para)
            current_words += pw

    if current:
        packed.append("\n\n".join(current))

    return _apply_word_overlap(packed, overlap_words)


def _body_chunks(content: str, hp: Dict[str, Any]) -> List[str]:
    mode = str(hp_get(hp, "chunking.mode", "word_windows")).lower()
    overlap_words = int(hp_get(hp, "chunking.overlap_words", 35))

    if mode == "paragraph":
        max_chunk_words = int(hp_get(hp, "chunking.max_chunk_words", 400))
        return _paragraph_chunks(
            content,
            max_chunk_words=max_chunk_words,
            overlap_words=overlap_words,
        )

    chunk_words = int(hp_get(hp, "chunking.chunk_words", 140))
    return _word_chunks(
        content,
        chunk_words=chunk_words,
        overlap_words=overlap_words,
    )


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """
    Split one corpus entry into retrieval units.

    Modes (chunking.mode):
    - word_windows: fixed-size word chunks with overlap
    - paragraph: merge paragraphs up to max_chunk_words, then word overlap
    """
    page_id = int(record["page_id"])
    title = str(record.get("title", "")).strip()
    content = str(record.get("content", "")).strip()

    hp = load_hparams()
    title_chunk_enabled = bool(hp_get(hp, "chunking.title_chunk", True))

    chunks: List[Chunk] = []
    if title_chunk_enabled and title:
        chunks.append(
            Chunk(
                page_id=page_id,
                chunk_id=-1,
                text=format_chunk_text(title, title),
            )
        )

    content_chunks = _body_chunks(content, hp)
    for i, body in enumerate(content_chunks):
        body = body.strip()
        if title:
            text = format_chunk_text(title, body)
        else:
            text = format_chunk_text("", body) if body else body
        chunks.append(Chunk(page_id=page_id, chunk_id=i, text=text))

    if not chunks:
        fallback = entry_text(record)
        chunks = [
            Chunk(
                page_id=page_id,
                chunk_id=0,
                text=format_chunk_text(title, fallback) if title else fallback,
            )
        ]
    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
