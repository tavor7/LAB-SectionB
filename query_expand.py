"""Query expansion helpers (stdlib only)."""
from __future__ import annotations

from lexical import tokenize

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "then",
        "than",
        "too",
        "very",
        "just",
        "about",
        "into",
        "over",
        "after",
        "before",
        "between",
        "under",
        "again",
        "further",
        "once",
    }
)


def keyword_query(query: str) -> str:
    """Lowercase query with simple English stopwords removed."""
    tokens = [t for t in tokenize(query) if t not in STOPWORDS]
    if not tokens:
        return query.lower().strip()
    return " ".join(tokens)


def query_versions(query: str, *, use_expansion: bool) -> tuple[str, str]:
    """Return (original, keyword) query strings."""
    original = query.strip()
    if not use_expansion:
        return original, original
    return original, keyword_query(original)
