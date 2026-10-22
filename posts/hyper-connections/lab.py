#!/usr/bin/env python3
"""Hyper-Connections: independent NumPy checks, float64, no model downloads.
Run: python lab.py
Axes: X[old_stream, channel], C[old_stream, new_stream].
"""
import numpy as np


def layer_norm(x, eps=1e-5):
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + eps)


def layer_norm_jvp(x, dx, eps=1e-5):
    centered = x - x.mean(axis=-1, keepdims=True)
    dc = dx - dx.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + eps)
    return dc / rms - centered * np.mean(centered * dc, axis=-1, keepdims=True) / rms**3


def hc(x, p, q, c, w):
    return c.T @ x + q[:, None] * np.tanh((p @ x) @ w)


def static_jvp(x, dx, p, q, c, w):
    f = np.tanh((p @ x) @ w)
    df = (1 - f**2) * ((p @ dx) @ w)
    return c.T @ dx + q[:, None] * df


def dynamic(x, params, dx=None):
    base_alpha, base_q, wa, wb, w, scale = params
    normalized = layer_norm(x)
    ta, tb = np.tanh(normalized @ wa), np.tanh(normalized @ wb)
    alpha, q = base_alpha + scale * ta, base_q + scale * tb
    p, c = alpha[:, 0], alpha[:, 1:]
    f = np.tanh((p @ x) @ w)
    out = c.T @ x + q[:, None] * f
    if dx is None:
        return out
    dn = layer_norm_jvp(x, dx)
    da, dq = scale * (1 - ta**2) * (dn @ wa), scale * (1 - tb**2) * (dn @ wb)
    dp, dc = da[:, 0], da[:, 1:]
    df = (1 - f**2) * (((p @ dx) + (dp @ x)) @ w)
    derivative = c.T @ dx + dc.T @ x + dq[:, None] * f + q[:, None] * df
    return out, derivative


def check_derivative(name, function, analytic, x, dx):
    eps = 1e-6
    finite = (function(x + eps * dx) - function(x - eps * dx)) / (2 * eps)
    error = np.linalg.norm(finite - analytic) / max(np.linalg.norm(finite), 1e-12)
    print(f"{name} directional derivative relative error: {error:.3e}")
    assert error < 1e-6


def main():
    rng = np.random.default_rng(42)
    n, d = 3, 5
    x, dx = rng.normal(size=(n, d)), rng.normal(size=(n, d))
    p, q, c, w = rng.normal(size=n), rng.normal(size=n), rng.normal(size=(n, n)), rng.normal(size=(d, d)) / 3
    actual = hc(x, p, q, c, w)
    # Explicit scalar loops form an independent reference for all axes.
    z = np.array([sum(p[i] * x[i, k] for i in range(n)) for k in range(d)])
    f = np.tanh(np.array([sum(z[k] * w[k, j] for k in range(d)) for j in range(d)]))
    expected = np.array([[sum(c[i, j] * x[i, k] for i in range(n)) + q[j] * f[k]
                          for k in range(d)] for j in range(n)])
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    # Stream relabeling must commute with the HC operation.
    permutation = np.eye(n)[[2, 0, 1]]
    changed = hc(permutation @ x, permutation @ p, permutation @ q, permutation @ c @ permutation.T, w)
    np.testing.assert_allclose(changed, permutation @ actual, atol=1e-12)
    print("explicit summation and stream permutation: passed")

    # At initialization, each replicated stream follows the same residual model.
    h = rng.normal(size=d)
    expanded = np.tile(h, (n, 1))
    for layer in range(8):
        weights = rng.normal(size=(d, d)) / 4
        read = np.eye(n)[layer % n]
        h = h + np.tanh(layer_norm(h) @ weights)
        z = read @ expanded
        expanded = expanded + np.ones((n, 1)) * np.tanh(layer_norm(z) @ weights)
        np.testing.assert_allclose(expanded, np.tile(h, (n, 1)), atol=1e-12)
    np.testing.assert_allclose(expanded.mean(axis=0), h, atol=1e-12)
    print("replicated initialization over 8 sublayers: passed")

    check_derivative("static", lambda v: hc(v, p, q, c, w), static_jvp(x, dx, p, q, c, w), x, dx)
    params = (np.concatenate([p[:, None], c], axis=1), q,
              rng.normal(size=(d, n + 1)) / 2, rng.normal(size=d) / 2, w, 0.03)
    _, derivative = dynamic(x, params, dx)
    check_derivative("dynamic (all gate paths)", lambda v: dynamic(v, params), derivative, x, dx)
    # Analytic two-stream path weight from the exercise.
    mix = np.array([[.8, .2], [.2, .8]])
    contribution = np.array([0., 1.]) @ mix.T @ mix.T @ np.array([1., 0.])
    assert abs(contribution - .32) < 1e-12
    print(f"two-step historical contribution: {contribution:.2f}")
    print("all checks passed")


if __name__ == "__main__":
    main()
