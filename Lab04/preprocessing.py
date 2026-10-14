"""Small, leakage-safe text inspection helpers for Lab 04."""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any


_TOKEN_PATTERN = re.compile(r"[\w]+(?:['’][\w]+)*", flags=re.UNICODE)


def normalize_text(text: Any) -> str:
    """Normalize Unicode and case, retaining words such as ``not`` and ``don't``."""
    if is_missing_text(text):
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).lower().replace("’", "'")
    return " ".join(_TOKEN_PATTERN.findall(normalized))


def tokenize_text(text: Any) -> list[str]:
    """Return normalized word tokens without removing negation."""
    return _TOKEN_PATTERN.findall(normalize_text(text))


def is_missing_text(text: Any) -> bool:
    """Return whether one text value is null, NaN, or blank after trimming."""
    if text is None:
        return True
    if isinstance(text, float) and math.isnan(text):
        return True
    return not str(text).strip()


def has_missing_text(texts: Iterable[Any]) -> bool:
    """Return whether an iterable contains at least one missing text value."""
    return any(is_missing_text(text) for text in texts)


def document_length_statistics(texts: Iterable[Any]) -> dict[str, int | float]:
    """Summarize documents by whitespace-token count; empty documents have length 0."""
    lengths = [len(str(text).split()) if not is_missing_text(text) else 0 for text in texts]
    if not lengths:
        return {"count": 0, "average": 0.0, "median": 0, "min": 0, "max": 0}
    return {
        "count": len(lengths),
        "average": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "min": min(lengths),
        "max": max(lengths),
    }


def validate_labels(labels: Sequence[Any], allowed_labels: set[Any] | None = None) -> dict[Any, int]:
    """Validate nonmissing labels and optionally reject values outside an allowed set."""
    counts: dict[Any, int] = {}
    for label in labels:
        if label is None or (isinstance(label, float) and math.isnan(label)) or not str(label).strip():
            raise ValueError("Labels must not be missing or blank")
        if allowed_labels is not None and label not in allowed_labels:
            raise ValueError(f"Unexpected label: {label!r}")
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        raise ValueError("At least one label is required")
    return counts


def sigmoid(value: float) -> float:
    """Numerically stable logistic sigmoid used by the lab's hand calculation."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)
