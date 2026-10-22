#!/usr/bin/env python3
"""Sparse-upcycling CPU probes. Python 3 + NumPy; all examples synthetic.

Run: python3 lab.py
Checks gates and router derivatives, expert symmetry breaking, exact SwiGLU
partitioning, output scaling, and logical parameter/communication budgets.
"""
import numpy as np


def route(logits, values, topk, normalize):
    """Scalar expert outputs; derivative holds away from top-k crossings."""
    chosen = np.argsort(logits)[-topk:]
    if normalize:
        p = np.zeros_like(logits)
        local = np.exp(logits[chosen] - np.max(logits[chosen]))
        p[chosen] = local / local.sum()
        gate = p
        output = gate @ values
        derivative = gate * (values - output)
    else:
        p = np.exp(logits - np.max(logits))
        p /= p.sum()
        selected_values = np.zeros_like(values)
        selected_values[chosen] = values[chosen]
        gate = np.zeros_like(p)
        gate[chosen] = p[chosen]
        output = gate @ values
        derivative = p * (selected_values - output)
    return output, gate, derivative


def finite_gradient(fn, x, h=1e-6):
    result = np.empty_like(x)
    for index in np.ndindex(x.shape):
        a, b = x.copy(), x.copy()
        a[index] += h
        b[index] -= h
        result[index] = (fn(a) - fn(b)) / (2 * h)
    return result


def toy_loss(experts, router, inputs, targets):
    """Linear experts E×d and router E×d, selected-softmax top-2."""
    de, dr, loss = np.zeros_like(experts), np.zeros_like(router), 0.
    for x, target in zip(inputs, targets):
        y, gate, dy = route(router @ x, experts @ x, 2, True)
        error = y - target
        loss += .5 * error**2
        de += error * np.outer(gate, x)
        dr += error * np.outer(dy, x)
    return loss / len(inputs), de / len(inputs), dr / len(inputs)


def swiglu(x, gate, up, down):
    a = gate @ x
    hidden = (a / (1 + np.exp(-a))) * (up @ x)
    return down @ hidden


def main():
    z = np.array([1., 0., -1.])
    for normalize in [False, True]:
        for topk in [1, 2]:
            for values in [np.array([2., 2., 2.]), np.array([2., 3., 1.])]:
                y, gate, grad = route(z, values, topk, normalize)
                numeric = finite_gradient(lambda v: route(v, values, topk, normalize)[0], z)
                np.testing.assert_allclose(grad, numeric, atol=4e-10)
                if normalize and (topk == 1 or np.all(values == values[0])):
                    np.testing.assert_allclose(grad, 0, atol=1e-14)
                if normalize and np.all(values == values[0]):
                    np.testing.assert_allclose(y, values[0])
    y, _, grad = route(z, np.full(3, 2.), 2, False)
    print(f"Cloned experts, global-softmax top-2: y={y:.6f}; dy/dz={np.round(grad,6)}")
    y, _, grad = route(z, np.full(3, 2.), 2, True)
    print(f"Cloned experts, selected-softmax top-2: y={y:.6f}; router derivative norm={np.linalg.norm(grad):.6f}")

    inputs = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 1.]])
    targets = np.array([2., -1., 0., 3.])
    experts = np.tile([.5, -.25], (3, 1))
    router = np.array([[1., 0.], [0., 1.], [-1., -1.]])
    before, de, dr = toy_loss(experts, router, inputs, targets)
    np.testing.assert_allclose(dr, 0, atol=1e-14)
    experts -= .1 * de
    after, de, dr = toy_loss(experts, router, inputs, targets)
    assert after < before and np.linalg.norm(dr) > 1e-3
    numeric = finite_gradient(lambda v: toy_loss(experts, v, inputs, targets)[0], router)
    np.testing.assert_allclose(dr, numeric, atol=5e-10)
    print(f"One expert update: toy loss {before:.6f} -> {after:.6f}; router gradient norm now {np.linalg.norm(dr):.6f}")

    gate = np.array([[1., 0.], [0., -1.], [.5, 0.], [.5, -.5]])
    up = np.array([[1., 0.], [0., 1.], [1., .5], [1., -1.]])
    down = np.array([[1., .5, -.5, 1.], [.5, 1., 1., -.5]])
    rng = np.random.default_rng(4)
    for x in rng.normal(size=(20, 2)):
        full = swiglu(x, gate, up, down)
        parts = [swiglu(x, gate[s], up[s], down[:, s]) for s in [slice(0, 2), slice(2, 4)]]
        np.testing.assert_allclose(sum(parts), full, atol=2e-15)
        # One replica from each shard, gate=1/2, down-projection multiplier=2.
        restored = sum(.5 * 2 * part for part in parts)
        np.testing.assert_allclose(restored, full, atol=2e-15)
    x = np.array([1., -1.])
    full = swiglu(x, gate, up, down)
    a = swiglu(x, gate[:2], up[:2], down[:, :2])
    b = swiglu(x, gate[2:], up[2:], down[:, 2:])
    wrong = 2 * a
    assert np.linalg.norm(wrong - full) > .5
    # Three projections scaled by s do not make SiLU exactly homogeneous.
    scaled = swiglu(x, 2 * gate, 2 * up, 2 * down) / 8
    assert np.linalg.norm(scaled - full) > .1
    print(f"SwiGLU shards: A={np.round(a,6)}, B={np.round(b,6)}, full={np.round(full,6)}")
    print(f"Two copies of A after scale correction: {np.round(wrong,6)}; output error={np.linalg.norm(wrong-full):.6f}")
    print(f"Scaling all SwiGLU matrices then dividing by 8: error={np.linalg.norm(scaled-full):.6f}")

    shared, dense_ffn, total_experts, granularity, topk = 1e9, 6e9, 64, 8, 8
    total = shared + total_experts / granularity * dense_ffn
    active = shared + topk / granularity * dense_ffn
    np.testing.assert_allclose([total / 1e9, active / 1e9], [49, 7])
    assert 2 * 1024 * 1 * 4096 * 2 / 2**20 == 16
    assert 2 * 1024 * 8 * 4096 * 2 / 2**20 == 128
    print(f"Parameter budget (router excluded): total={total/1e9:.0f}B active={active/1e9:.0f}B")
    print("Coarse/fine equal-FLOP case: tokens per expert 128/128; logical payload 16/128 MiB per layer")
    print("PASS: router finite differences, symmetry breaking, SwiGLU reconstruction, scale and budget checks")


if __name__ == '__main__':
    main()
