#!/usr/bin/env python3
"""mHC: numerical contracts and finite-iteration derivatives, NumPy only.
Run: python lab.py
C[i,j] maps old stream i to new stream j; the state update is C.T @ X.
All tests are CPU numerical tests, not training/throughput benchmarks.
"""
import numpy as np


def logsumexp(x, axis):
    maximum = x.max(axis=axis, keepdims=True)
    return maximum + np.log(np.exp(x - maximum).sum(axis=axis, keepdims=True))


def log_sinkhorn(logits, rounds=20, direction=None):
    """Normalize rows then columns; optionally return the exact program JVP."""
    log_c = np.array(logits, dtype=np.float64, copy=True)
    derivative = None if direction is None else np.array(direction, dtype=np.float64, copy=True)
    for _ in range(rounds):
        for axis in (1, 0):
            normalizer = logsumexp(log_c, axis)
            if derivative is not None:
                weights = np.exp(log_c - normalizer)
                derivative -= (weights * derivative).sum(axis=axis, keepdims=True)
            log_c -= normalizer
    c = np.exp(log_c)
    return c if derivative is None else (c, c * derivative)


def v4_style_fp32(logits, rounds=20, eps=1e-6):
    """NumPy reference for the published V4 inference normalization order.
    This reproduces coefficient arithmetic, not GPU kernel performance.
    """
    c = np.array(logits, dtype=np.float32, copy=True)
    c = np.exp(c - c.max(axis=1, keepdims=True))
    c = c / c.sum(axis=1, keepdims=True) + np.float32(eps)
    c /= c.sum(axis=0, keepdims=True) + np.float32(eps)
    for _ in range(rounds - 1):
        c /= c.sum(axis=1, keepdims=True) + np.float32(eps)
        c /= c.sum(axis=0, keepdims=True) + np.float32(eps)
    return c


def errors(c):
    return np.max(abs(c.sum(axis=1) - 1)), np.max(abs(c.sum(axis=0) - 1))


def main():
    # Independent hand-derived one-round answer.
    actual = log_sinkhorn(np.log([[4., 1.], [1., 1.]]), rounds=1)
    expected = np.array([[8 / 13, 2 / 7], [5 / 13, 5 / 7]])
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert abs(errors(actual)[0] - 9 / 91) < 1e-14
    print("one-round rational reference: passed")

    mild = np.log([[8., 1., 1.], [1., 3., 1.], [2., 1., 1.]])
    sharp = np.array([[10., 0., -10.], [10., 0., -10.], [-10., 0., 10.]])
    for name, logits in [("mild", mild), ("sharp", sharp)]:
        c = log_sinkhorn(logits, rounds=20)
        row, col = errors(c)
        v4 = v4_style_fp32(logits)
        print(f"{name}, 20 rounds: row error={row:.6g}, column error={col:.3e}")
        print(f"  V4-style FP32 + epsilon: row/column errors={errors(v4)}")
        assert np.linalg.norm(c, 2) <= np.sqrt(c.sum(0).max() * c.sum(1).max()) + 1e-12
    assert errors(log_sinkhorn(sharp, 20))[0] > .03
    assert errors(log_sinkhorn(mild, 20))[0] < 1e-10
    assert np.isfinite(log_sinkhorn(sharp * 100, 20)).all()

    rng = np.random.default_rng(7)
    c = log_sinkhorn(rng.normal(size=(4, 4)), rounds=200)
    other = log_sinkhorn(rng.normal(size=(4, 4)), rounds=200)
    x = rng.normal(size=(4, 9))
    np.testing.assert_allclose((c.T @ x).mean(0), x.mean(0), atol=1e-13)
    assert np.linalg.norm(c.T @ x) <= np.linalg.norm(x) + 1e-12
    assert max(errors(c @ other)) < 1e-12
    assert abs(np.linalg.norm(c, 2) - 1) < 1e-12
    print("mean conservation, non-expansion and product closure: passed")

    # A row-stochastic counterexample: total sum survives, L2 norm grows.
    row_only = np.array([[1., 0.], [1., 0.]])
    value = np.ones(2)
    mixed = row_only.T @ value
    assert mixed.sum() == value.sum()
    assert np.linalg.norm(mixed) > np.linalg.norm(value)
    print(f"row-only norm ratio: {np.linalg.norm(mixed) / np.linalg.norm(value):.6f}")

    a, value = .9, np.array([2., 6.])
    mixing = np.array([[a, 1 - a], [1 - a, a]])
    for depth in range(41):
        closed = 4 + (2 * a - 1)**depth * np.array([-2., 2.])
        np.testing.assert_allclose(value, closed, atol=1e-12)
        value = mixing.T @ value
    print("two-stream contrast closed form through 40 layers: passed")

    # Exact derivative of the *finite* normalization program vs central difference.
    logits, direction = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    _, derivative = log_sinkhorn(logits, rounds=5, direction=direction)
    eps = 1e-6
    finite = (log_sinkhorn(logits + eps * direction, 5) - log_sinkhorn(logits - eps * direction, 5)) / (2 * eps)
    relative = np.linalg.norm(derivative - finite) / np.linalg.norm(finite)
    assert relative < 1e-6
    print(f"finite Sinkhorn JVP relative error: {relative:.3e}")

    # Scalar normalization can move past the linear projection.
    v, w = rng.normal(size=(1, 24)), rng.normal(size=(24, 8))
    rms = np.sqrt(np.mean(v**2, axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose((v / rms) @ w, (v @ w) / rms, atol=1e-13)
    # Uniform pre and unit post exactly recover the residual when streams agree.
    h = rng.normal(size=9)
    expanded = np.tile(h, (4, 1))
    output = c.T @ expanded + np.ones((4, 1)) * np.tanh(expanded.mean(axis=0))
    np.testing.assert_allclose(output, np.tile(h + np.tanh(h), (4, 1)), atol=1e-13)
    print("projection reordering and replicated initialization: passed")
    print("all checks passed")


if __name__ == "__main__":
    main()
