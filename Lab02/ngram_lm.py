"""A small, dependency-free unigram/bigram/trigram language model.

The implementation uses no sentence-boundary markers. Each input sentence is
counted independently, so an n-gram never crosses a sentence boundary. A
sentence score starts with a unigram probability, then uses the largest
available context up to the model order. This convention gives the lab's
``the cat eats fish`` bigram example a probability of 1/24.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any


UNK = "<UNK>"
_WORD_RE = re.compile(r"<UNK>|[\w]+(?:['’][\w]+)*", re.IGNORECASE | re.UNICODE)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")


def _normalize_token(token: Any) -> str:
    """Normalize one already-tokenized word while preserving the UNK marker."""
    value = unicodedata.normalize("NFKC", str(token)).strip()
    if value.casefold() == UNK.casefold():
        return UNK
    return value.casefold()


def tokenize(text: str) -> list[str]:
    """Lowercase and extract word-like tokens, retaining contractions/numbers."""
    normalized = unicodedata.normalize("NFKC", text)
    return [
        UNK if token.casefold() == UNK.casefold() else token.casefold()
        for token in _WORD_RE.findall(normalized)
    ]


def split_sentences(text: str) -> list[str]:
    """Split raw text on terminal punctuation followed by whitespace/newlines."""
    return [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(text) if part.strip()]


def iter_tokenized_sentences(text: str) -> Iterator[list[str]]:
    """Yield non-empty tokenized sentences from a raw document."""
    for sentence in split_sentences(text):
        tokens = tokenize(sentence)
        if tokens:
            yield tokens


def _iter_sentences(corpus: Iterable[Any] | str) -> Iterator[list[str]]:
    """Accept sentence strings or token sequences and skip empty sentences."""
    if isinstance(corpus, str):
        corpus = (corpus,)
    for sentence in corpus:
        if isinstance(sentence, str):
            tokens = tokenize(sentence)
        else:
            try:
                tokens = [_normalize_token(token) for token in sentence]
            except TypeError as exc:
                raise TypeError("Each sentence must be text or a token sequence") from exc
        tokens = [token for token in tokens if token]
        if tokens:
            yield tokens


def _reiterable(corpus: Iterable[Any] | str) -> Iterable[Any]:
    """Materialize one-shot iterators so vocabulary and counts use same input."""
    if isinstance(corpus, str):
        return (corpus,)
    iterator = iter(corpus)
    if iterator is corpus:
        return tuple(iterator)
    return corpus


def build_vocabulary(
    corpus: Iterable[Any] | str, unk_token: str | None = None
) -> dict[str, int]:
    """Build a deterministic vocabulary; ``<UNK>`` is optional by default."""
    terms = {token for sentence in _iter_sentences(corpus) for token in sentence}
    if unk_token is not None:
        terms.add(unk_token)
    return {word: index for index, word in enumerate(sorted(terms))}


def _validate_order(n: int) -> None:
    if n not in (1, 2, 3):
        raise ValueError("n must be 1, 2, or 3")


def count_ngrams(corpus: Iterable[Any] | str, n: int) -> Counter:
    """Count observed n-grams only; unigrams use string keys, higher orders tuples."""
    _validate_order(n)
    counts: Counter = Counter()
    for sentence in _iter_sentences(corpus):
        if n == 1:
            counts.update(sentence)
            continue
        counts.update(tuple(sentence[i : i + n]) for i in range(len(sentence) - n + 1))
    return counts


def perplexity_from_log_probabilities(log_probabilities: Iterable[float]) -> float:
    """Compute exp(-mean(log p)) without multiplying sentence probabilities."""
    values = list(log_probabilities)
    if not values:
        return math.nan
    mean_log_probability = math.fsum(values) / len(values)
    if mean_log_probability == -math.inf:
        return math.inf
    exponent = -mean_log_probability
    if exponent > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
        return math.inf
    return math.exp(exponent)


class NGramLanguageModel:
    """Maximum-likelihood or add-one n-gram model for n in ``{1, 2, 3}``.

    ``fit`` expects a re-iterable collection of sentence strings/token lists,
    or a one-shot iterator (which is materialized). N-grams do not cross
    sentence boundaries. For conditional counts, a history is counted only
    when it has a following token; this keeps distributions normalized without
    adding an explicit ``</s>`` event.
    """

    def __init__(
        self,
        n: int,
        smoothing: str | None = None,
        unk_token: str | None = UNK,
    ) -> None:
        _validate_order(n)
        if smoothing is not None:
            normalized_smoothing = smoothing.casefold().replace("_", "-")
            if normalized_smoothing not in {"laplace", "add-one", "addone"}:
                raise ValueError("smoothing must be None or 'laplace'/'add-one'")
            smoothing = "laplace"
        self.n = n
        self.smoothing = smoothing
        self.unk_token = unk_token
        self.vocabulary: dict[str, int] = {}
        self.ngram_counts: dict[int, Counter] = {}
        self.context_counts: dict[int, Counter] = {}
        self.token_count = 0
        self.is_fitted = False

    def fit(self, corpus: Iterable[Any] | str) -> NGramLanguageModel:
        """Fit the vocabulary and sparse observed n-gram counters on corpus."""
        source = _reiterable(corpus)
        self.vocabulary = build_vocabulary(source, unk_token=self.unk_token)
        self.ngram_counts = {order: Counter() for order in range(1, self.n + 1)}
        self.token_count = 0

        for sentence in _iter_sentences(source):
            mapped = [self._map_token(token) for token in sentence]
            mapped = [token for token in mapped if token is not None]
            self.token_count += len(mapped)
            for order in range(1, self.n + 1):
                if order == 1:
                    self.ngram_counts[order].update(mapped)
                else:
                    self.ngram_counts[order].update(
                        tuple(mapped[i : i + order])
                        for i in range(len(mapped) - order + 1)
                    )

        self.context_counts = {}
        for order in range(2, self.n + 1):
            contexts: Counter = Counter()
            for ngram, frequency in self.ngram_counts[order].items():
                contexts[ngram[:-1]] += frequency
            self.context_counts[order] = contexts
        self.is_fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Call fit(corpus) before scoring or predicting")

    def _map_token(self, token: str) -> str | None:
        normalized = _normalize_token(token)
        if normalized in self.vocabulary:
            return normalized
        if self.unk_token is not None and self.unk_token in self.vocabulary:
            return self.unk_token
        return None

    def _input_tokens(self, value: str | Sequence[str] | None) -> list[str]:
        if value is None:
            return []
        tokens = tokenize(value) if isinstance(value, str) else [_normalize_token(t) for t in value]
        return [token for token in tokens if token]

    def _effective_order(self, context: Sequence[str]) -> int:
        return min(self.n, len(context) + 1)

    def _counts_for(self, context: Sequence[str]) -> tuple[int, int, int]:
        """Return (order, n-gram count, history count) after context backoff."""
        order = self._effective_order(context)
        word_counts = self.ngram_counts.get(order, Counter())
        if order == 1:
            target_key: Any = None
            history_count = self.token_count
        else:
            history = tuple(context[-(order - 1) :])
            target_key = history
            history_count = self.context_counts.get(order, Counter())[history]
        return order, history_count, target_key

    def probability(self, context: str | Sequence[str] | None, word: str) -> float:
        """Return MLE or Laplace probability of ``word`` after ``context``.

        Contexts shorter than ``n - 1`` back off to the largest available
        shorter order. An unobserved word maps to ``<UNK>`` when enabled.
        """
        self._require_fitted()
        # Only the most recent n-1 tokens can affect an n-gram probability.
        # Trimming before normalization avoids repeatedly mapping every token
        # in long sentence prefixes during log scoring and perplexity.
        if self.n == 1 or context is None:
            raw_context = []
        elif isinstance(context, str):
            raw_context = self._input_tokens(context)[-(self.n - 1) :]
        else:
            try:
                recent_context = context[-(self.n - 1) :]
            except (TypeError, KeyError):
                recent_context = context
            raw_context = self._input_tokens(recent_context)[-(self.n - 1) :]
        mapped_context = [self._map_token(token) for token in raw_context]
        if any(token is None for token in mapped_context):
            return 0.0
        mapped_word = self._map_token(word)
        if mapped_word is None:
            return 0.0

        context_tokens = [token for token in mapped_context if token is not None]
        order, history_count, history = self._counts_for(context_tokens)
        if order == 1:
            observed_count = self.ngram_counts[1][mapped_word]
        else:
            observed_count = self.ngram_counts[order][(*history, mapped_word)]

        if self.smoothing == "laplace":
            vocabulary_size = len(self.vocabulary)
            if vocabulary_size == 0:
                return 0.0
            return (observed_count + 1) / (history_count + vocabulary_size)
        if history_count == 0:
            return 0.0
        return observed_count / history_count

    def _log_probability(self, context: Sequence[str], word: str) -> float:
        probability = self.probability(context, word)
        return math.log(probability) if probability > 0.0 else -math.inf

    def sentence_log_probability(self, sentence: str | Sequence[str]) -> float:
        """Sum per-token log probabilities directly, returning ``-inf`` for zero."""
        self._require_fitted()
        tokens = self._input_tokens(sentence)
        total = 0.0
        for index, word in enumerate(tokens):
            context_start = max(0, index - (self.n - 1))
            log_probability = self._log_probability(tokens[context_start:index], word)
            if log_probability == -math.inf:
                return -math.inf
            total += log_probability
        return total

    def sentence_probability(self, sentence: str | Sequence[str]) -> float:
        """Return sentence probability, underflowing to 0 only at float limits."""
        log_probability = self.sentence_log_probability(sentence)
        if log_probability == -math.inf:
            return 0.0
        if log_probability < math.log(float.fromhex("0x0.0000000000001p-1022")):
            return 0.0
        return math.exp(log_probability)

    def next_word_distribution(self, context: str | Sequence[str] | None) -> dict[str, float]:
        """Return probabilities for every vocabulary item, including ``<UNK>``."""
        self._require_fitted()
        context_tokens = self._input_tokens(context)
        probabilities = {
            word: self.probability(context_tokens, word)
            for word in sorted(self.vocabulary)
        }
        return probabilities

    def predict_next(
        self, context: str | Sequence[str] | None, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Return the highest-probability next words, with lexical tie-breaking."""
        if top_k <= 0:
            return []
        distribution = self.next_word_distribution(context)
        ranked = sorted(
            ((word, probability) for word, probability in distribution.items() if probability > 0.0),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[:top_k]

    def perplexity(self, corpus: Iterable[Any] | str) -> float:
        """Compute token-level perplexity over sentence events in ``corpus``."""
        self._require_fitted()
        total_log_probability = 0.0
        token_count = 0
        saw_zero = False
        for tokens in _iter_sentences(corpus):
            for index, word in enumerate(tokens):
                context_start = max(0, index - (self.n - 1))
                log_probability = self._log_probability(tokens[context_start:index], word)
                token_count += 1
                if log_probability == -math.inf:
                    saw_zero = True
                elif not saw_zero:
                    total_log_probability += log_probability
        if token_count == 0:
            return math.nan
        if saw_zero:
            return math.inf
        exponent = -total_log_probability / token_count
        if exponent > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
            return math.inf
        return math.exp(exponent)

    def rank_candidates(
        self,
        context: str | Sequence[str] | None,
        candidates: Iterable[str | Sequence[str]],
    ) -> list[dict[str, Any]]:
        """Score candidate continuations and sort by average log probability."""
        return rank_candidates(context, candidates, self)


def _history_key(tokens: Sequence[str], order: int) -> Any:
    return tokens[-(order - 1) :] if order > 1 else []


def rank_candidates(
    context: str | Sequence[str] | None,
    candidates: Iterable[str | Sequence[str]],
    model: NGramLanguageModel,
) -> list[dict[str, Any]]:
    """Rank candidate text continuations by average log probability.

    Each result includes both total (raw) and per-token (average) log scores.
    Empty candidates receive a score of zero and are ranked after finite scores.
    """
    model._require_fitted()
    context_tokens = model._input_tokens(context)
    results = []
    for candidate in candidates:
        candidate_tokens = model._input_tokens(candidate)
        history = list(context_tokens)
        score = 0.0
        zero_probability = False
        for word in candidate_tokens:
            value = model._log_probability(history, word)
            if value == -math.inf:
                zero_probability = True
                break
            score += value
            history.append(word)
        if zero_probability:
            score = -math.inf
        average = score / len(candidate_tokens) if candidate_tokens else 0.0
        display_candidate = candidate if isinstance(candidate, str) else " ".join(candidate)
        results.append(
            {
                "candidate": display_candidate,
                "raw_log_probability": score,
                "average_log_probability": average,
                "token_count": len(candidate_tokens),
            }
        )

    def sort_key(item: Mapping[str, Any]) -> tuple[float, str]:
        average = item["average_log_probability"]
        if item["token_count"] == 0:
            return math.inf, item["candidate"]
        return (-average if math.isfinite(average) else math.inf, item["candidate"])

    results.sort(key=sort_key)
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    return results


def unseen_ngram_statistics(
    model: NGramLanguageModel, corpus: Iterable[Any] | str
) -> dict[str, int]:
    """Count distinct unseen n-gram types/events and zero-probability events."""
    model._require_fitted()
    eval_counts: Counter = Counter()
    zero_probability_events = 0
    for tokens in _iter_sentences(corpus):
        mapped = [model._map_token(token) for token in tokens]
        mapped = [token for token in mapped if token is not None]
        eval_counts.update(count_ngrams([mapped], model.n))
        for index, word in enumerate(mapped):
            context_start = max(0, index - (model.n - 1))
            if model.probability(mapped[context_start:index], word) == 0.0:
                zero_probability_events += 1
    unseen = {
        ngram: count
        for ngram, count in eval_counts.items()
        if ngram not in model.ngram_counts[model.n]
    }
    return {
        "unseen_ngram_types": len(unseen),
        "unseen_ngram_events": sum(unseen.values()),
        "zero_probability_events": zero_probability_events,
    }


def _train(n: int, corpus: Iterable[Any] | str, **kwargs: Any) -> NGramLanguageModel:
    return NGramLanguageModel(n, **kwargs).fit(corpus)


def train_unigram(corpus: Iterable[Any] | str, **kwargs: Any) -> NGramLanguageModel:
    """Convenience function to train an unigram model."""
    return _train(1, corpus, **kwargs)


def train_bigram(corpus: Iterable[Any] | str, **kwargs: Any) -> NGramLanguageModel:
    """Convenience function to train a bigram model."""
    return _train(2, corpus, **kwargs)


def train_trigram(corpus: Iterable[Any] | str, **kwargs: Any) -> NGramLanguageModel:
    """Convenience function to train a trigram model."""
    return _train(3, corpus, **kwargs)


def probability(model: NGramLanguageModel, context: str | Sequence[str] | None, word: str) -> float:
    """Functional wrapper around ``NGramLanguageModel.probability``."""
    return model.probability(context, word)


def sentence_probability(model: NGramLanguageModel, sentence: str | Sequence[str]) -> float:
    """Functional wrapper around ``NGramLanguageModel.sentence_probability``."""
    return model.sentence_probability(sentence)


def sentence_log_probability(model: NGramLanguageModel, sentence: str | Sequence[str]) -> float:
    """Functional wrapper around ``NGramLanguageModel.sentence_log_probability``."""
    return model.sentence_log_probability(sentence)


__all__ = [
    "NGramLanguageModel",
    "UNK",
    "build_vocabulary",
    "count_ngrams",
    "iter_tokenized_sentences",
    "perplexity_from_log_probabilities",
    "probability",
    "rank_candidates",
    "sentence_log_probability",
    "sentence_probability",
    "split_sentences",
    "tokenize",
    "train_bigram",
    "train_trigram",
    "train_unigram",
    "unseen_ngram_statistics",
]
