#!/usr/bin/env python3
"""CPU NumPy experiments for shared depth. Run: python lab.py"""
import argparse
import numpy as np


def scalar_forward(a, e, rounds):
    states = [0.0]
    for _ in range(rounds):
        states.append(a * states[-1] + e)
    return np.array(states)


def scalar_gradient(a, e, rounds, target, window):
    states = scalar_forward(a, e, rounds)
    grad_state = states[-1] - target
    grad_a, grad_e = 0.0, 0.0
    for r in range(rounds, max(0, rounds-window), -1):
        grad_a += grad_state * states[r-1]
        grad_e += grad_state
        grad_state *= a
    return np.array([grad_a, grad_e])


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=-1, keepdims=True)


def parallel_depth(e, wq, wk, wv, rounds, share_first=False):
    """Teacher-forced causal attention, one sequence-wide call per round."""
    z = e.copy()
    first_k, first_v = e @ wk, e @ wv
    for _ in range(rounds):
        q = z @ wq
        k, v = (first_k, first_v) if share_first else (z @ wk, z @ wv)
        logits = q @ k.T / np.sqrt(e.shape[1])
        logits[np.triu_indices(e.shape[0], 1)] = -np.inf
        z = np.tanh(e + softmax(logits) @ v)
    return z


def cached_depth(e, wq, wk, wv, rounds, share_first=False):
    """Token-major decode; cache slot r stores projections of round r input."""
    keys, values = [[] for _ in range(rounds)], [[] for _ in range(rounds)]
    result = []
    for embedding in e:
        z = embedding.copy()
        for r in range(rounds):
            q = z @ wq
            slot = 0 if share_first else r
            if not share_first or r == 0:
                keys[slot].append(z @ wk)
                values[slot].append(z @ wv)
            logits = np.array(keys[slot]) @ q / np.sqrt(e.shape[1])
            z = np.tanh(embedding + softmax(logits) @ np.array(values[slot]))
        result.append(z)
    return np.array(result)


def main():
    a, e, rounds, target = .5, 1., 3, 2.
    states = scalar_forward(a, e, 8)
    np.testing.assert_allclose(states, e*(1-a**np.arange(9))/(1-a), atol=1e-12)
    print('PASS shared iterations: z3 = %.6f; z8 = %.6f' % (states[3], states[8]))

    full = scalar_gradient(a, e, rounds, target, rounds)
    truncated = scalar_gradient(a, e, rounds, target, 1)
    epsilon = 1e-6
    def objective(aa, ee):
        return .5*(scalar_forward(aa, ee, rounds)[-1]-target)**2
    numeric = np.array([(objective(a+epsilon, e)-objective(a-epsilon, e))/(2*epsilon),
                        (objective(a, e+epsilon)-objective(a, e-epsilon))/(2*epsilon)])
    np.testing.assert_allclose(full, numeric, atol=1e-9)
    np.testing.assert_allclose(full[0], -.5, atol=1e-12)
    np.testing.assert_allclose(truncated[0], -.375, atol=1e-12)
    # Detach the boundary: finite difference now changes only the final call.
    boundary = scalar_forward(a, e, rounds-1)[-1]
    fixed_boundary_loss = lambda aa: .5*(aa*boundary+e-target)**2
    numeric_trunc = (fixed_boundary_loss(a+epsilon)-fixed_boundary_loss(a-epsilon))/(2*epsilon)
    np.testing.assert_allclose(truncated[0], numeric_trunc, atol=1e-9)
    print('PASS gradient: full da = %.6f; truncated da = %.6f' % (full[0], truncated[0]))

    rng = np.random.default_rng(20261022)
    embeddings = rng.normal(size=(7, 4))
    weights = [rng.normal(size=(4, 4))*.5 for _ in range(3)]
    outputs = []
    for share_first in (False, True):
        dense = parallel_depth(embeddings, *weights, 3, share_first)
        cached = cached_depth(embeddings, *weights, 3, share_first)
        np.testing.assert_allclose(dense, cached, atol=1e-12)
        # Future suffixes never change an earlier causal output.
        prefix = parallel_depth(embeddings[:4], *weights, 3, share_first)
        np.testing.assert_allclose(dense[:4], prefix, atol=1e-12)
        outputs.append(dense)
        print('PASS KV protocol %s: dense/decode max error %.2e' %
              ('first-round' if share_first else 'per-round', np.max(np.abs(dense-cached))))
    difference = np.max(np.abs(outputs[0]-outputs[1]))
    assert difference > 1e-3
    print('PASS protocol changes computation: max output difference %.6f' % difference)

    budgets = np.array([1, 3, 2, 3, 1, 3])
    assert budgets.sum() == 13
    for mode in ('full', 'routed', 'shared'):
        keys = np.arange(1, 7, dtype=float)
        if mode == 'routed':
            keys = keys[budgets >= 3]
        if mode != 'shared':
            keys *= 3
        output = softmax(.25*keys) @ keys
        print('figure, %s: final attention output %.6f' % (mode, output))
    cache_bytes = 2*2*4096*6*3*4*64*2
    assert cache_bytes//2**20 == 144
    assert cache_bytes//6//2**20 == 24
    work = np.array([1, 2, 3, 6])
    assert work.sum() == 12 and len(work)*work.max() == 24
    print('PASS budgets: cache 144 / 24 MiB; useful work 12 / 24 slots')
    print('All numerical checks passed.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-a', type=float, default=.5, help='retention in an extra 8-round experiment')
    args = parser.parse_args()
    curve = scalar_forward(args.experiment_a, 1., 8)
    print('experiment: a=%.3f states=%s' % (args.experiment_a, np.array2string(curve, precision=5)))
    print('experiment: da(full,last-round) = %.6f, %.6f' %
          (scalar_gradient(args.experiment_a, 1., 8, 2., 8)[0],
           scalar_gradient(args.experiment_a, 1., 8, 2., 1)[0]))


if __name__ == '__main__':
    main()
