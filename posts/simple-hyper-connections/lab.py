#!/usr/bin/env python3
"""SimpleHC: exact grouped execution, gradient paths, and traffic accounting.

Run: python lab.py
Dependency: NumPy (python -m pip install numpy)
Source: https://arxiv.org/abs/2611.05984v1, Sections 3-4 / Appendix B.

Each row of X is one stream; columns are channels. Each column of P/Q
contains one sublayer's read/write gates. These CPU float64 experiments use
small tanh branches for checking algebra; benchmark results require the
paper's trained models and GPU kernels.
"""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gates(x, w_pre, w_post, b_pre, b_post, eps=1e-6):
    """Independent layer gates share the normalized n*C entry state."""
    flat = x.reshape(-1)
    inv_rms = 1.0 / np.sqrt(np.mean(flat * flat) + eps)
    z = flat * inv_rms
    p = sigmoid(np.einsum('knm,m->kn', w_pre, z) + b_pre).T
    q = 2.0 * sigmoid(np.einsum('knm,m->kn', w_post, z) + b_post).T
    return p, q, (flat, inv_rms, z)


def sequential(x, p, q, weights, biases):
    """Materialize the n*C state after every sublayer, using given gates."""
    state = x.copy()
    inputs, outputs = [], []
    for j in range(p.shape[1]):
        u = state.T @ p[:, j]
        o = np.tanh(weights[j] @ u + biases[j])
        state += q[:, j, None] * o[None, :]
        inputs.append(u)
        outputs.append(o)
    return state, np.column_stack(inputs), np.column_stack(outputs)


def grouped(x, p, q, weights, biases, corrections=True):
    """Algorithm 1: base reads + causal scalar corrections + one writeback."""
    width, count = x.shape[1], p.shape[1]
    bases = x.T @ p                         # C x K
    coeff = np.tril(p.T @ q, -1)            # K x K: row=reader, col=writer
    inputs = np.empty((width, count))
    outputs = np.empty((width, count))
    for j in range(count):
        inputs[:, j] = bases[:, j]
        if corrections and j:
            inputs[:, j] += outputs[:, :j] @ coeff[j, :j]
        outputs[:, j] = np.tanh(weights[j] @ inputs[:, j] + biases[j])
    state = x + q @ outputs.T
    return state, inputs, outputs, coeff


def full_forward(params):
    x, wp, wq, bp, bq, weights, biases = params
    p, q, gate_cache = gates(x, wp, wq, bp, bq)
    state, inputs, outputs, coeff = grouped(x, p, q, weights, biases)
    return state, (p, q, gate_cache, inputs, outputs, coeff)


def full_backward(params, output_grad, cache):
    """Analytic reverse pass, including D, tanh, sigmoid, and RMSNorm paths."""
    x, wp, wq, bp, bq, weights, biases = params
    p, q, (flat, inv_rms, z), inputs, outputs, coeff = cache
    count = p.shape[1]

    # Y = X + Q O^T. E = dLoss/dY.
    dx = output_grad.copy()
    dq = output_grad @ outputs
    do = output_grad.T @ q
    dv = np.zeros_like(inputs)
    dd = np.zeros_like(coeff)
    dw = np.zeros_like(weights)
    db = np.zeros_like(biases)

    # Reverse the sequential nonlinear sublayers and all correction edges.
    for j in reversed(range(count)):
        dt = do[:, j] * (1.0 - outputs[:, j] ** 2)
        du = weights[j].T @ dt
        dw[j] = np.outer(dt, inputs[:, j])
        db[j] = dt
        dv[:, j] = du
        for previous in range(j):
            dd[j, previous] = du @ outputs[:, previous]
            do[:, previous] += coeff[j, previous] * du

    # V = X^T P; D = tril(P^T Q, -1).
    dx += p @ dv.T
    dp = x @ dv + q @ dd.T
    dq += p @ dd

    # P = sigmoid(logits_pre), Q = 2 sigmoid(logits_post).
    dlp = (dp * p * (1.0 - p)).T
    dlq = (dq * q * (1.0 - q / 2.0)).T
    dwp = dlp[:, :, None] * z
    dwq = dlq[:, :, None] * z
    dz = np.einsum('knm,kn->m', wp, dlp)
    dz += np.einsum('knm,kn->m', wq, dlq)

    # z = x / sqrt(mean(x^2) + eps), affine-free RMSNorm.
    dx_gate = inv_rms * dz - flat * (inv_rms ** 3 / flat.size) * (dz @ flat)
    dx += dx_gate.reshape(x.shape)
    return (dx, dwp, dwq, dlp, dlq, dw, db)


def dynamic_sequential(params):
    """identityHC reference: each gate conditions on its current state."""
    x, wp, wq, bp, bq, weights, biases = params
    state = x.copy()
    for j in range(weights.shape[0]):
        p, q, _ = gates(state, wp[j:j+1], wq[j:j+1], bp[j:j+1], bq[j:j+1])
        u = state.T @ p[:, 0]
        o = np.tanh(weights[j] @ u + biases[j])
        state += q[:, 0, None] * o[None, :]
    return state


def traffic_per_width(n, count):
    """Forward residual activation elements / C / sublayer (Eq. 10)."""
    wide = 4.0 * n / count
    narrow = 2.0
    correction = (count - 1.0) * (count + 4.0) / (2.0 * count)
    return wide, narrow, correction


def visual_example():
    x = np.array([2.0, 6.0])
    p = np.array([[.75, .25, .5, .8], [.25, .75, .5, .2]])
    q = np.array([[1., .5, 1., .25], [.5, 1., 1., .75]])
    bases = x @ p
    coeff = np.tril(p.T @ q, -1)
    outputs = np.zeros(4)
    states = [x.copy()]
    print('Hand-calculable example: two streams, C=1, K=4, F(u)=u/2')
    for j in range(4):
        correction = coeff[j, :j] @ outputs[:j]
        u = bases[j] + correction
        outputs[j] = u / 2.0
        state = states[-1] + q[:, j] * outputs[j]
        np.testing.assert_allclose(u, states[-1] @ p[:, j], rtol=1e-13, atol=1e-13)
        states.append(state)
        print(f'  sublayer {j}: input={bases[j]:.6f}+{correction:.6f}={u:.6f}; output={outputs[j]:.6f}')
    grouped_exit = x + q @ outputs
    np.testing.assert_allclose(grouped_exit, states[-1], rtol=1e-13, atol=1e-13)
    print('  correction matrix D =')
    print(coeff)
    print('  group exit =', grouped_exit)
    np.testing.assert_allclose(outputs, [1.5, 2.96875, 3.67578125, 4.803515625])
    np.testing.assert_allclose(grouped_exit, [9.86103515625, 16.99716796875])


def main():
    np.set_printoptions(precision=8, suppress=True)
    rng = np.random.default_rng(20261111)
    n, width, count = 4, 7, 4
    x = rng.normal(size=(n, width))
    wp = rng.normal(scale=.15, size=(count, n, n * width))
    wq = rng.normal(scale=.15, size=(count, n, n * width))
    bp = rng.normal(scale=.2, size=(count, n))
    bq = rng.normal(scale=.2, size=(count, n))
    weights = rng.normal(scale=.2, size=(count, width, width))
    biases = rng.normal(scale=.1, size=(count, width))
    params = (x, wp, wq, bp, bq, weights, biases)
    actual, cache = full_forward(params)
    p, q = cache[:2]
    reference, ref_inputs, ref_outputs = sequential(x, p, q, weights, biases)
    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache[3], ref_inputs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache[4], ref_outputs, rtol=1e-12, atol=1e-12)
    print('Same shared-entry gates: grouped vs materialized sequential execution')
    print(f'  max output error = {np.max(np.abs(actual - reference)):.3e}')

    without, *_ = grouped(x, p, q, weights, biases, corrections=False)
    reused_p = np.repeat(p[:, :1], count, axis=1)
    reused_q = np.repeat(q[:, :1], count, axis=1)
    reused, *_ = grouped(x, reused_p, reused_q, weights, biases)
    current_state = dynamic_sequential(params)
    print('Toy output differences (max absolute difference from shared-entry model):')
    for label, changed in [('remove scalar corrections', without),
                           ('reuse the first gate pair', reused),
                           ('condition gates on each current state', current_state)]:
        difference = np.max(np.abs(actual - changed))
        assert difference > 1e-6, f'{label}: toy should expose the changed function'
        print(f'  {label}: {difference:.6f}')

    output_grad = rng.normal(size=actual.shape)
    gradients = full_backward(params, output_grad, cache)
    names = ('entry state X', 'read gate weights', 'write gate weights',
             'read gate bias', 'write gate bias', 'branch weights', 'branch bias')
    epsilon = 1e-6
    print('Directional derivative checks, including gate dependence on X:')
    for index, (name, value, grad) in enumerate(zip(names, params, gradients)):
        direction = rng.normal(size=value.shape)
        direction /= np.linalg.norm(direction)
        plus, minus = list(params), list(params)
        plus[index], minus[index] = value + epsilon * direction, value - epsilon * direction
        finite_difference = np.sum((full_forward(tuple(plus))[0] - full_forward(tuple(minus))[0]) * output_grad) / (2.0 * epsilon)
        analytic = np.sum(grad * direction)
        relative_error = abs(analytic - finite_difference) / max(1.0, abs(analytic), abs(finite_difference))
        assert relative_error < 2e-7, (name, analytic, finite_difference, relative_error)
        print(f'  {name}: relative error = {relative_error:.3e}')

    print('Logical forward residual activation traffic, n=4:')
    print('  K | wide/C + narrow/C + correction/C = total/C | saved entry states')
    for group in (1, 2, 4, 5, 8, 16):
        wide, narrow, correction = traffic_per_width(4, group)
        saved = 1.0 - 1.0 / group
        print(f'  {group:2d} | {wide:6.3f} + {narrow:.3f} + {correction:6.3f} = {wide+narrow+correction:6.3f} | {saved:6.2%}')
    np.testing.assert_allclose([sum(traffic_per_width(4, k)) for k in (1, 2, 4, 5, 8, 16)],
                               [18, 11.5, 9, 8.8, 9.25, 12.375])
    print('  K=5 minimizes this formula over integers 1..16; paper uses K=4.')
    print('  Saved fraction applies to residual entry states; other activations remain.')
    visual_example()
    print('all checks passed')


if __name__ == '__main__':
    main()
