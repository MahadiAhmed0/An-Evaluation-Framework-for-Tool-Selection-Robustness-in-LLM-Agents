"""Unit tests for the embedding-based tool retriever.

Tests use a deterministic bag-of-words fake embedder (no sentence-transformers
download required) to verify ranking, similarity math, metric validation, and
document-embedding caching. This module evaluates defensive retrieval
benchmarking only; it generates no attack content.
"""

from typing import Callable, List, Tuple

import numpy as np
import pytest

from tool_selection_harness.core import Retriever, ToolDocument, ToolLibrary

STOPWORDS = {
    "the", "a", "an", "for", "in", "is", "of", "to", "and", "or",
    "what", "with", "please", "into",
}


def _tokens(text: str) -> list[str]:
    """Lowercase, strip punctuation, and filter stopwords/short tokens."""
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    return [w for w in cleaned.split() if len(w) > 2 and w not in STOPWORDS]


class BagEmbedder:
    """Deterministic one-hot bag-of-words embedder with call counting."""

    def __init__(self, vocab: list[str]) -> None:
        self._index = {word: i for i, word in enumerate(vocab)}
        self.call_count = 0
        self.seen_texts: list[str] = []

    def __call__(self, text: str) -> np.ndarray:
        self.call_count += 1
        self.seen_texts.append(text)
        vec = np.zeros(len(self._index), dtype=np.float32)
        for word in _tokens(text):
            idx = self._index.get(word)
            if idx is not None:
                vec[idx] = 1.0
        return vec


DOCS = {
    "weather_tool": "Fetch current weather conditions city",
    "calendar_tool": "Add calendar event tomorrow meeting",
    "translate_text": "Translate text between languages spanish french",
    "calculate": "Evaluate mathematical expression result number",
}

QUERIES = {
    "weather": "what is the weather forecast in paris",
    "calendar": "add calendar event tomorrow",
}


def _doc_text(name: str, description: str) -> str:
    return f"{name}: {description}"


@pytest.fixture
def vocab() -> list[str]:
    corpus = [_doc_text(n, d) for n, d in DOCS.items()]
    corpus.extend(QUERIES.values())
    return sorted({w for text in corpus for w in _tokens(text)})


@pytest.fixture
def embedder(vocab: list[str]) -> BagEmbedder:
    return BagEmbedder(vocab)


@pytest.fixture
def retriever(embedder: BagEmbedder) -> Retriever:
    return Retriever(embed_fn=embedder)


@pytest.fixture
def library() -> ToolLibrary:
    return ToolLibrary(
        documents=[ToolDocument(name, desc) for name, desc in DOCS.items()]
    )