"""Core attention implementation for LAB 06 (NumPy only).

Contains a numerically stable softmax, (scaled) dot-product attention,
self-attention and a few helpers (entropy, Jensen-Shannon distance).
No torch / keras attention layers are used here. Run this file to execute the
hand-calculation tests:  python attention_numpy.py
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def softmax(x: Array, axis: int = -1, temperature: float = 1.0) -> Array:
    """Numerically stable softmax(x / temperature) along ``axis``.

    Subtracting max(x) leaves the result unchanged because
    exp(x_i - m) / sum_j exp(x_j - m) = exp(x_i) exp(-m) / (exp(-m) sum_j exp(x_j)).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    z = np.asarray(x, dtype=np.float64) / temperature
    z = z - np.max(z, axis=axis, keepdims=True)  # largest entry becomes 0 -> exp <= 1
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def entropy(p: Array, axis: int = -1) -> Array | float:
    """Shannon entropy in nats; 0*log(0) is treated as 0."""
    p = np.asarray(p, dtype=np.float64)
    safe = np.where(p > 0, p, 1.0)  # log(1)=0 -> zero-probability terms vanish
    return -np.sum(p * np.log(safe), axis=axis)


def jensen_shannon(p: Array, q: Array) -> float:
    """Jensen-Shannon divergence (nats) between two distributions; in [0, ln 2]."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return float(entropy(m) - 0.5 * entropy(p) - 0.5 * entropy(q))


def total_variation(p: Array, q: Array) -> float:
    """Total variation distance 0.5 * sum |p - q|, in [0, 1]."""
    return float(0.5 * np.sum(np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64))))


def _check_qkv(Q: Array, K: Array, V: Array) -> tuple[Array, Array, Array]:
    """Validate shapes: Q (..., T_q, d_k), K (..., T_k, d_k), V (..., T_k, d_v)."""
    Q, K, V = (np.asarray(a, dtype=np.float64) for a in (Q, K, V))
    if Q.ndim < 2 or K.ndim < 2 or V.ndim < 2:
        raise ValueError(f"Q, K, V must be at least 2-D, got {Q.shape}, {K.shape}, {V.shape}")
    if not (Q.ndim == K.ndim == V.ndim):
        raise ValueError(f"Q, K, V must have the same number of dims, got {Q.ndim}, {K.ndim}, {V.ndim}")
    if Q.shape[:-2] != K.shape[:-2] or K.shape[:-2] != V.shape[:-2]:
        raise ValueError(f"batch dims differ: {Q.shape[:-2]}, {K.shape[:-2]}, {V.shape[:-2]}")
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError(f"d_k mismatch: Q has {Q.shape[-1]}, K has {K.shape[-1]}")
    if K.shape[-2] != V.shape[-2]:
        raise ValueError(f"number of keys ({K.shape[-2]}) != number of values ({V.shape[-2]})")
    return Q, K, V


def _attend(Q: Array, K: Array, V: Array, scale: bool, mask: Array | None) -> tuple[Array, Array, Array]:
    Q, K, V = _check_qkv(Q, K, V)
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2)  # (..., T_q, T_k)
    if scale:
        scores = scores / np.sqrt(d_k)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        try:
            np.broadcast_shapes(mask.shape, scores.shape)
        except ValueError as exc:
            raise ValueError(f"mask shape {mask.shape} not broadcastable to scores {scores.shape}") from exc
        # True = keep, False = block (weight exactly 0 after softmax)
        scores = np.where(mask, scores, -1e30)
    weights = softmax(scores, axis=-1)  # rows sum to 1
    context = weights @ V  # (..., T_q, d_v)
    return scores, weights, context


def scaled_dot_product_attention(
    Q: Array, K: Array, V: Array, mask: Array | None = None
) -> tuple[Array, Array, Array]:
    """Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V.

    Args:
        Q: (T_q, d_k) or (B, T_q, d_k) queries.
        K: (T_k, d_k) or (B, T_k, d_k) keys.
        V: (T_k, d_v) or (B, T_k, d_v) values.
        mask: optional bool array broadcastable to (T_q, T_k); True = may attend.
    Returns:
        (scores, attention_weights, context) with shapes
        (T_q, T_k), (T_q, T_k), (T_q, d_v) (plus batch dim if given).
        ``scores`` are the scaled scores before softmax (masked entries = -1e30).
    """
    return _attend(Q, K, V, scale=True, mask=mask)


def dot_product_attention(
    Q: Array, K: Array, V: Array, mask: Array | None = None
) -> tuple[Array, Array, Array]:
    """Same as above but WITHOUT the 1/sqrt(d_k) scaling (for comparison)."""
    return _attend(Q, K, V, scale=False, mask=mask)


def self_attention(
    X: Array, W_Q: Array, W_K: Array, W_V: Array, mask: Array | None = None
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Self-attention: Q = X W_Q, K = X W_K, V = X W_V.

    X: (T, d_model); W_Q, W_K: (d_model, d_k); W_V: (d_model, d_v).
    Returns (Q, K, V, scores, attention_weights, context).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim < 2:
        raise ValueError(f"X must be at least 2-D (T, d_model), got {X.shape}")
    for name, W in (("W_Q", W_Q), ("W_K", W_K), ("W_V", W_V)):
        W = np.asarray(W)
        if W.ndim != 2 or W.shape[0] != X.shape[-1]:
            raise ValueError(f"{name} must be (d_model={X.shape[-1]}, d), got {W.shape}")
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    scores, weights, context = scaled_dot_product_attention(Q, K, V, mask)
    return Q, K, V, scores, weights, context


def _run_tests() -> None:
    atol = 1e-4
    ref = np.array([0.09003, 0.24473, 0.66524])

    # Numerical stability (section 17)
    big = softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert np.all(np.isfinite(big)) and np.isclose(big.sum(), 1.0)
    assert np.allclose(big, ref, atol=atol)
    assert np.allclose(softmax(np.array([1.0, 2.0, 3.0])), ref, atol=atol)

    # Independent reference (tests only)
    from scipy.special import softmax as sp_softmax
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 7)) * 50
    assert np.allclose(softmax(x, axis=-1), sp_softmax(x, axis=-1), atol=1e-12)
    assert np.allclose(softmax(x, axis=0), sp_softmax(x, axis=0), atol=1e-12)
    assert np.allclose(softmax(x, temperature=2.0), sp_softmax(x / 2.0, axis=-1), atol=1e-12)
    try:
        import torch
        assert np.allclose(softmax(x), torch.softmax(torch.tensor(x), dim=-1).numpy(), atol=1e-12)
    except ImportError:
        pass

    # Problem 4: no scaling
    q = np.array([[1.0, 0.0]])
    K = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    V = np.array([[1, 0], [0, 2], [2, 1]], dtype=float)
    s, a, c = dot_product_attention(q, K, V)
    assert np.allclose(s, [[1, 0, 1]], atol=atol)
    assert np.allclose(a, [[0.42232, 0.15536, 0.42232]], atol=atol)
    assert np.allclose(c, [[1.26696, 0.73304]], atol=atol)

    # Problem 3: context vector
    alpha = np.array([0.1, 0.2, 0.7])
    V3 = np.array([[1, 0], [0, 1], [2, 2]], dtype=float)
    assert np.allclose(alpha @ V3, [1.5, 1.6], atol=atol)

    # Self-attention example (sections 20-22)
    X = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    W_Q = np.array([[1, 1], [0, 1]], dtype=float)
    W_K = np.array([[1, 0], [1, 1]], dtype=float)
    W_V = np.array([[1, 0], [0, 2]], dtype=float)
    Q, Kk, Vv, S, A, C = self_attention(X, W_Q, W_K, W_V)
    A_ref = np.array([[0.14003, 0.28400, 0.57598], [0.19778, 0.40111, 0.40111], [0.07432, 0.30570, 0.61999]])
    C_ref = np.array([[0.7160, 1.7199], [0.5989, 1.6044], [0.6943, 1.8514]])
    assert np.allclose(A, A_ref, atol=atol)
    assert np.allclose(C, C_ref, atol=atol)
    assert np.allclose(A.sum(axis=1), 1.0)

    # Batch dimension + mask + shape errors
    Qb = rng.normal(size=(2, 4, 8)); Kb = rng.normal(size=(2, 5, 8)); Vb = rng.normal(size=(2, 5, 3))
    sb, ab, cb = scaled_dot_product_attention(Qb, Kb, Vb)
    assert sb.shape == (2, 4, 5) and ab.shape == (2, 4, 5) and cb.shape == (2, 4, 3)
    for i in range(2):
        assert np.allclose(scaled_dot_product_attention(Qb[i], Kb[i], Vb[i])[2], cb[i])
    causal = np.tril(np.ones((4, 4), dtype=bool))
    _, am, _ = scaled_dot_product_attention(rng.normal(size=(4, 8)), rng.normal(size=(4, 8)), rng.normal(size=(4, 3)), causal)
    assert np.allclose(np.triu(am, 1), 0.0) and np.allclose(am.sum(-1), 1.0)
    for bad in (lambda: scaled_dot_product_attention(np.ones((2, 3)), np.ones((2, 4)), np.ones((2, 1))),
                lambda: scaled_dot_product_attention(np.ones((2, 3)), np.ones((2, 3)), np.ones((5, 1))),
                lambda: scaled_dot_product_attention(np.ones(3), np.ones((2, 3)), np.ones((2, 1)))):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    # Entropy / divergence sanity
    assert np.isclose(entropy(np.array([0.5, 0.5, 0.0])), np.log(2))
    assert np.isclose(jensen_shannon(np.array([1.0, 0.0]), np.array([0.0, 1.0])), np.log(2))
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    _run_tests()
