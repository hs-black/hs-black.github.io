#!/usr/bin/env python3
"""CPU NumPy checks for state-space models. Run: python lab.py.
The tiny operators verify identities and state contracts, not language-model quality.
"""
import numpy as np


def recur(a, b, c, x, initial=None):
    n, p = b.shape[1], x.shape[1]
    state = np.zeros((n, p)) if initial is None else initial.copy()
    outputs = []
    for at, bt, ct, xt in zip(a, b, c, x):
        state = at * state + np.outer(bt, xt)
        outputs.append(ct @ state)
    return np.stack(outputs), state


def dense(a, b, c, x):
    length = len(a)
    matrix = np.zeros((length, length))
    for t in range(length):
        for j in range(t + 1):
            matrix[t, j] = np.prod(a[j + 1:t + 1]) * (c[t] @ b[j])
    return matrix @ x


def chunked(a, b, c, x, chunk):
    state = np.zeros((b.shape[1], x.shape[1]))
    output = []
    for lo in range(0, len(a), chunk):
        hi = min(lo + chunk, len(a))
        aa, bb, cc, xx = a[lo:hi], b[lo:hi], c[lo:hi], x[lo:hi]
        local = dense(aa, bb, cc, xx)
        incoming = (cc @ state) * np.cumprod(aa)[:, None]
        output.append(local + incoming)
        weights = np.array([np.prod(aa[j + 1:]) for j in range(len(aa))])
        state = np.prod(aa) * state + (bb * weights[:, None]).T @ xx
    return np.concatenate(output), state


def main():
    rng = np.random.default_rng(19)
    # General time-invariant matrix recurrence vs independently constructed convolution.
    matrix = np.array([[.5, .2], [-.1, .7]])
    b, c, x = np.array([1., .3]), np.array([.2, 1.]), rng.normal(size=15)
    state = np.zeros(2)
    ys = []
    for xt in x:
        state = matrix @ state + b * xt
        ys.append(c @ state)
    kernel = np.array([c @ np.linalg.matrix_power(matrix, i) @ b for i in range(len(x))])
    np.testing.assert_allclose(ys, np.convolve(x, kernel)[:len(x)], atol=1e-13)
    print('PASS fixed-system recurrence = explicit convolution')

    length, n, p = 19, 4, 3
    a = rng.uniform(.2, .95, length)
    b, c, x = rng.normal(size=(length, n)), rng.normal(size=(length, n)), rng.normal(size=(length, p))
    y, final = recur(a, b, c, x)
    np.testing.assert_allclose(y, dense(a, b, c, x), atol=1e-12)
    for size in (1, 4, 7, 19):
        actual, state = chunked(a, b, c, x, size)
        np.testing.assert_allclose(y, actual, atol=1e-12)
        np.testing.assert_allclose(final, state, atol=1e-12)
    first, middle = recur(a[:7], b[:7], c[:7], x[:7])
    second, _ = recur(a[7:], b[7:], c[7:], x[7:], middle)
    np.testing.assert_allclose(np.concatenate([first, second]), y, atol=1e-12)
    reset, _ = recur(a[7:], b[7:], c[7:], x[7:])
    packed_a = a.copy(); packed_a[7] = 0
    packed, _ = recur(packed_a, b, c, x)
    np.testing.assert_allclose(packed[7:], reset, atol=1e-12)
    assert np.max(np.abs(reset - second)) > 1e-3
    print('PASS SSD dense/chunk/streaming paths and packed sequence reset')

    # Differentiate an independently expanded scalar recurrence in alpha.
    values, alpha = np.array([1., 0., 2., 0., -1., 1.]), .5
    def final_state(aa):
        h = 0.
        for val in values:
            h = aa * h + val
        return h
    powers = np.arange(len(values) - 1, -1, -1)
    gradient = sum(v * k * alpha**(k-1) for v, k in zip(values, powers) if k)
    delta = 1e-6
    numerical = (final_state(alpha+delta)-final_state(alpha-delta))/(2*delta)
    np.testing.assert_allclose(gradient, numerical, rtol=1e-8)
    print('PASS scalar decay gradient; final state:', final_state(alpha))

    # Two-endpoint write expanded into a lower triangular matrix.
    length = 8
    aa, beta, gamma, u = (rng.uniform(.1, .9, length) for _ in range(4))
    h, outputs = 0., []
    for t in range(length):
        h = aa[t] * h + gamma[t] * u[t] + (beta[t] * u[t-1] if t else 0.)
        outputs.append(h)
    mask = np.zeros((length, length))
    for t in range(length):
        for j in range(t+1):
            mask[t, j] = gamma[j] * np.prod(aa[j+1:t+1])
            if j < t:
                mask[t, j] += beta[j+1] * np.prod(aa[j+2:t+1])
    np.testing.assert_allclose(outputs, mask @ u, atol=1e-13)
    print('PASS exponential-trapezoidal recurrence = two-band/decay expansion')

    # Complex recurrence vs a rotating coordinate frame, with arbitrary writes/readouts.
    angles, rho = rng.normal(size=12), .9
    writes = rng.normal(size=(12, 2)); reads = rng.normal(size=(12, 2))
    raw, transformed, cumulative = np.zeros(2), np.zeros(2), 0.
    def rotation(angle):
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for angle, write, read in zip(angles, writes, reads):
        raw = rho * rotation(angle) @ raw + write
        cumulative += angle
        frame = rotation(-cumulative)
        transformed = rho * transformed + frame @ write
        np.testing.assert_allclose(read @ raw, (frame @ read) @ transformed, atol=1e-12)
    print('PASS complex-state rotation = rotated input/readout coordinates')

    state = rng.normal(size=(5, 4)); b = rng.normal(size=(5, 3)); x = rng.normal(size=(4, 3)); c = rng.normal(size=(5, 3))
    updated = .8 * state + b @ x.T
    reference = .8 * state + sum(np.outer(b[:, r], x[:, r]) for r in range(3))
    np.testing.assert_allclose(updated, reference, atol=1e-13)
    np.testing.assert_allclose(updated.T @ c, np.column_stack([updated.T @ c[:, r] for r in range(3)]))
    print('PASS MIMO update = sum of rank-one writes')
    zoh = np.exp(-1) + 3*(1-np.exp(-1))/2
    ee = np.exp(-1) + 1.5
    print(f'Exercise 1: ZOH={zoh:.6f}, exponential-Euler={ee:.6f}, forward-Euler=1.500000')
    print('All numerical checks passed.')


if __name__ == '__main__':
    main()
