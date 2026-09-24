"""Core TF-IDF implementation and reproducible sparse search experiments.

The small-corpus functions implement the lab equations directly. The corpus
experiments use SciPy sparse matrices and scikit-learn only for count indexing.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize


def _tokens(document: str | Sequence[str]) -> list[str]:
    """Return whitespace tokens for a string, or a copy of a token sequence."""
    return document.split() if isinstance(document, str) else list(document)


def build_vocabulary(documents: Sequence[str | Sequence[str]]) -> dict[str, int]:
    """Build an alphabetically ordered vocabulary with deterministic indices."""
    terms = sorted({term for document in documents for term in _tokens(document)})
    return {term: index for index, term in enumerate(terms)}


def compute_counts(
    documents: Sequence[str | Sequence[str]], vocabulary: dict[str, int]
) -> np.ndarray:
    """Return dense count rows in vocabulary order (intended for small corpora)."""
    counts = np.zeros((len(documents), len(vocabulary)), dtype=np.int64)
    for row, document in enumerate(documents):
        for term, count in Counter(_tokens(document)).items():
            if term in vocabulary:
                counts[row, vocabulary[term]] = count
    return counts


def compute_tf(counts: np.ndarray) -> np.ndarray:
    """Compute raw term frequency: count divided by document token count."""
    values = np.asarray(counts, dtype=float)
    lengths = values.sum(axis=1, keepdims=True)
    return np.divide(values, lengths, out=np.zeros_like(values), where=lengths != 0)


def compute_idf(counts: np.ndarray) -> np.ndarray:
    """Compute unsmoothed IDF: log(N / df), as specified in the lab."""
    values = np.asarray(counts)
    n_documents = values.shape[0]
    if n_documents == 0:
        return np.zeros(values.shape[1], dtype=float)
    document_frequency = np.count_nonzero(values, axis=0)
    return np.log(np.divide(n_documents, document_frequency,
                            out=np.ones(values.shape[1], dtype=float),
                            where=document_frequency != 0))


def compute_tfidf(tf: np.ndarray, idf: np.ndarray) -> np.ndarray:
    """Multiply TF values by their corresponding IDF values, without smoothing."""
    return np.asarray(tf, dtype=float) * np.asarray(idf, dtype=float)


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Compute cosine similarity; a zero vector has similarity zero."""
    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 5) -> float:
    """Fraction of the first k retrieved IDs that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(doc_id in relevant for doc_id in retrieved[:k]) / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 5) -> float:
    """Fraction of all labeled relevant IDs present in the first k results."""
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(doc_id in relevant for doc_id in retrieved[:k]) / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant result, or zero when absent."""
    return next((1.0 / rank for rank, doc_id in enumerate(retrieved, 1)
                 if doc_id in relevant), 0.0)


def mean_reciprocal_rank(reciprocal_ranks: Sequence[float]) -> float:
    """Mean of per-query reciprocal ranks (zero for an empty sequence)."""
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def load_corpus(path: str | Path) -> list[dict[str, str]]:
    """Read JSONL or JSONL.GZ records, assigning stable line-based IDs."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    documents = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text", "")
            if not isinstance(text, str):
                raise ValueError(f"Record {line_number} has no string 'text' field")
            documents.append({
                "document_id": f"D{line_number:05d}",
                "text": text,
                "url": str(record.get("url", "")),
                "timestamp": str(record.get("timestamp", "")),
            })
    return documents


def tokenize_minimal(text: str) -> list[str]:
    """Pipeline A: lowercase, then whitespace tokenization; punctuation remains attached."""
    return text.lower().split()


def normalize_punctuation(text: str) -> str:
    """Replace Unicode punctuation/control characters with spaces and lowercase text."""
    return "".join(" " if unicodedata.category(char)[0] in {"P", "C"} else char
                   for char in text.lower())


def tokenize_normalized(text: str) -> list[str]:
    """Pipeline B: normalize punctuation, split on whitespace, remove English stopwords."""
    return [token for token in normalize_punctuation(text).split()
            if token not in ENGLISH_STOP_WORDS]


def train_subword_tokenizer(texts: Iterable[str], vocab_size: int = 20_000):
    """Train a local WordPiece tokenizer; no pretrained model or download is used."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
        continuing_subword_prefix="##",
    )
    tokenizer.train_from_iterator((normalize_punctuation(t) for t in texts), trainer=trainer)
    return tokenizer


def tokenize_subword(text: str, tokenizer) -> list[str]:
    """Pipeline C: normalized punctuation followed by locally trained WordPiece."""
    return tokenizer.encode(normalize_punctuation(text)).tokens


def fit_sparse_index(tokenized_documents: Sequence[Sequence[str]]) -> dict:
    """Fit a count vocabulary and unsmoothed TF-IDF sparse index on documents."""
    vectorizer = CountVectorizer(analyzer=lambda tokens: tokens, lowercase=False,
                                 dtype=np.float64)
    counts = vectorizer.fit_transform(tokenized_documents).tocsr()
    n_documents = counts.shape[0]
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log(n_documents / df)
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    tf = sparse.diags(np.divide(1.0, lengths, out=np.zeros_like(lengths), where=lengths != 0)) @ counts
    matrix = (tf @ sparse.diags(idf)).tocsr()
    return {
        "vectorizer": vectorizer,
        "tfidf": matrix,
        "idf": idf,
        "df": df,
        "terms": vectorizer.get_feature_names_out(),
        "normalized_tfidf": normalize(matrix, norm="l2", axis=1, copy=True),
        "average_tokens": float(np.mean(lengths)),
        "sparsity": float(1.0 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
    }


def search_index(query: str, top_k: int, index: dict,
                 token_fn, records: Sequence[dict[str, str]]) -> list[dict]:
    """Transform a query with the fitted vocabulary and rank against its index."""
    query_tokens = token_fn(query)
    query_counts = index["vectorizer"].transform([query_tokens]).tocsr()
    lengths = np.asarray(query_counts.sum(axis=1)).ravel()
    query_tf = sparse.diags(np.divide(1.0, lengths, out=np.zeros_like(lengths), where=lengths != 0)) @ query_counts
    query_vector = (query_tf @ sparse.diags(index["idf"])).tocsr()
    query_vector = normalize(query_vector, norm="l2", axis=1)
    similarities = np.asarray((index["normalized_tfidf"] @ query_vector.T).toarray()).ravel()
    doc_indices = np.arange(len(records))
    order = np.lexsort((doc_indices, -similarities))[:top_k]
    return [{
        "rank": rank,
        "document_id": records[i]["document_id"],
        "similarity": float(similarities[i]),
        "preview": re.sub(r"\s+", " ", records[i]["text"]).strip()[:280],
        "url": records[i]["url"],
    } for rank, i in enumerate(order, 1)]


def query_oov_rate(query: str, index: dict, token_fn) -> float:
    """OOV fraction over query token units after that pipeline's tokenization."""
    tokens = token_fn(query)
    if not tokens:
        return 0.0
    vocabulary = index["vectorizer"].vocabulary_
    return sum(token not in vocabulary for token in tokens) / len(tokens)


def evaluate_pipeline(name: str, index: dict, token_fn,
                      records: Sequence[dict[str, str]], queries: Sequence[dict],
                      relevant_by_query: dict[str, set[str]], top_k: int = 5) -> tuple[list[dict], dict]:
    """Run ranked queries, return row-level results and aggregate metrics."""
    result_rows = []
    recalls, reciprocal_ranks = [], []
    for item in queries:
        results = search_index(item["query"], top_k, index, token_fn, records)
        relevant = relevant_by_query[item["query_id"]]
        ids = [result["document_id"] for result in results]
        p_at_k = precision_at_k(ids, relevant, top_k)
        r_at_k = recall_at_k(ids, relevant, top_k)
        rr = reciprocal_rank(ids, relevant)
        recalls.append(r_at_k)
        reciprocal_ranks.append(rr)
        for result in results:
            result_rows.append({
                "pipeline": name,
                "query_id": item["query_id"],
                "query": item["query"],
                "rank": result["rank"],
                "document_id": result["document_id"],
                "similarity": result["similarity"],
                "is_relevant": result["document_id"] in relevant,
                "precision_at_5": p_at_k,
                "recall_at_5": r_at_k,
                "reciprocal_rank": rr,
                "preview": result["preview"],
                "url": result["url"],
            })
    aggregate = {
        "pipeline": name,
        "vocabulary_size": int(index["tfidf"].shape[1]),
        "average_tokens_per_document": index["average_tokens"],
        "matrix_sparsity": index["sparsity"],
        "oov_rate": float(np.mean([query_oov_rate(q["query"], index, token_fn) for q in queries])),
        "precision_at_5": float(np.mean([
            precision_at_k([r["document_id"] for r in search_index(q["query"], top_k, index, token_fn, records)],
                           relevant_by_query[q["query_id"]], top_k) for q in queries
        ])),
        "recall_at_5": float(np.mean(recalls)),
        "mrr": mean_reciprocal_rank(reciprocal_ranks),
    }
    return result_rows, aggregate


def write_results_csv(rows: Sequence[dict], path: str | Path) -> None:
    """Write row-level retrieval results as UTF-8 CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["pipeline", "query_id", "query", "rank", "document_id", "similarity",
                  "is_relevant", "precision_at_5", "recall_at_5", "reciprocal_rank",
                  "preview", "url"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    import sys
    import unittest

    suite = unittest.defaultTestLoader.discover(
        str(Path(__file__).parent), pattern="test_implementation.py"
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
