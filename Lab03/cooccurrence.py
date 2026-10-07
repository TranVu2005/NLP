"""Sparse co-occurrence representations for the Lab 03 experiments."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, issparse


_TOKEN_RE = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")


def tokenize(text: str) -> list[str]:
    """Normalize text to lowercase word-like tokens."""
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


def iter_tokenized_sentences(corpus: Any) -> Iterable[list[str]]:
    """Yield token lists while keeping sentence/document boundaries intact.

    ``corpus`` may be one text string, an iterable of text documents/sentences,
    or an iterable of token sequences. Context windows never cross an item or
    sentence boundary.
    """
    if corpus is None:
        return
    if isinstance(corpus, str):
        items: Sequence[Any] = [corpus]
    else:
        try:
            items = list(corpus)
        except TypeError as exc:
            raise TypeError("corpus must be text or an iterable of text/token sequences") from exc

    for item in items:
        if isinstance(item, str):
            for sentence in _SENTENCE_RE.split(item):
                tokens = tokenize(sentence)
                if tokens:
                    yield tokens
        elif isinstance(item, Sequence):
            tokens = [str(token).casefold() for token in item if str(token).strip()]
            if tokens:
                yield tokens
        else:
            raise TypeError("each corpus item must be a string or token sequence")


def build_vocabulary(
    corpus: Iterable[str | Sequence[str]] | str,
    min_count: int = 1,
) -> tuple[dict[str, int], dict[int, str]]:
    """Build deterministic word-to-index and index-to-word mappings."""
    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    counts = Counter(token for sentence in iter_tokenized_sentences(corpus) for token in sentence)
    words = sorted(word for word, count in counts.items() if count >= min_count)
    word_to_index = {word: index for index, word in enumerate(words)}
    index_to_word = {index: word for word, index in word_to_index.items()}
    return word_to_index, index_to_word


def build_cooccurrence_matrix(
    corpus: Iterable[str | Sequence[str]] | str,
    vocabulary: Mapping[str, int],
    window_size: int,
) -> csr_matrix:
    """Build a symmetric sparse target-context count matrix."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    vocab_size = len(vocabulary)
    if sorted(vocabulary.values()) != list(range(vocab_size)):
        raise ValueError("vocabulary indices must be contiguous from 0")

    row_chunks: list[np.ndarray] = []
    column_chunks: list[np.ndarray] = []
    for sentence in iter_tokenized_sentences(corpus):
        indices = np.fromiter(
            (vocabulary.get(token, -1) for token in sentence),
            dtype=np.int32,
            count=len(sentence),
        )
        for offset in range(1, min(window_size, len(indices) - 1) + 1):
            left = indices[:-offset]
            right = indices[offset:]
            valid = (left >= 0) & (right >= 0)
            if valid.any():
                left_valid = left[valid]
                right_valid = right[valid]
                row_chunks.extend((left_valid, right_valid))
                column_chunks.extend((right_valid, left_valid))

    if not row_chunks:
        return csr_matrix((vocab_size, vocab_size), dtype=np.float64)
    rows = np.concatenate(row_chunks)
    columns = np.concatenate(column_chunks)
    values = np.ones(rows.size, dtype=np.int32)
    matrix = csr_matrix((values, (rows, columns)), shape=(vocab_size, vocab_size))
    matrix.sum_duplicates()
    return matrix


def _as_vector(value: Any) -> np.ndarray:
    if issparse(value):
        return np.asarray(value.toarray(), dtype=np.float64).ravel()
    return np.asarray(value, dtype=np.float64).ravel()


def cosine_similarity(x: Any, y: Any) -> float:
    """Compute cosine similarity, returning 0.0 when either vector is zero."""
    x_vector = _as_vector(x)
    y_vector = _as_vector(y)
    if x_vector.shape != y_vector.shape:
        raise ValueError("vectors must have the same shape")
    denominator = float(np.linalg.norm(x_vector) * np.linalg.norm(y_vector))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(x_vector, y_vector) / denominator)


def most_similar(
    word: str,
    matrix: Any,
    vocabulary: Mapping[str, int],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return the highest cosine-similarity neighbors, excluding ``word``."""
    word = word.casefold()
    if top_k <= 0 or word not in vocabulary:
        return []
    target_index = vocabulary.get(word)
    if target_index is None:
        return []
    target = matrix.getrow(target_index) if issparse(matrix) else np.asarray(matrix)[target_index]
    target_vector = _as_vector(target)
    target_norm = float(np.linalg.norm(target_vector))
    if target_norm == 0.0:
        return []

    if issparse(matrix):
        dots = np.asarray(matrix @ target_vector).ravel()
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    else:
        dense_matrix = np.asarray(matrix, dtype=np.float64)
        dots = dense_matrix @ target_vector
        norms = np.linalg.norm(dense_matrix, axis=1)
    denominators = norms * target_norm
    scores = np.divide(dots, denominators, out=np.zeros_like(dots, dtype=np.float64), where=denominators != 0)

    index_to_word = {index: token for token, index in vocabulary.items()}
    ranked = [
        (index_to_word[index], float(scores[index]))
        for index in range(len(vocabulary))
        if index != target_index and index in index_to_word
    ]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[:top_k]


__all__ = [
    "build_vocabulary",
    "build_cooccurrence_matrix",
    "cosine_similarity",
    "iter_tokenized_sentences",
    "most_similar",
    "tokenize",
]
