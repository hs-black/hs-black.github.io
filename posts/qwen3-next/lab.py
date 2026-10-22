#!/usr/bin/env python3
"""Qwen3-Next mechanisms on a CPU: NumPy is the only dependency.

    python -m pip install numpy
    python lab.py
    python lab.py --chunk 5 --batch 128 --conv-slots 4

The small hybrid below has a causal convolution, a Gated DeltaNet state,
and one attention cache. It tests the state protocol, using float64 arithmetic.
Memory counts use the published 80B-A3B configuration and explicit dtypes.
Timings and GPU throughput require a separate hardware measurement.
"""
from dataclasses import dataclass
import argparse
import numpy as np


def softmax(x, axis=-1):
    z = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return z / z.sum(axis=axis, keepdims=True)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def recurrent(q, k, v, alpha, beta, initial):
    """Article convention: S has shape [value_dim, key_dim]."""
    state = initial.copy()
    outputs = []
    for qt, kt, vt, a, b in zip(q, k, v, alpha, beta):
        state *= a
        state += b * np.outer(vt - state @ kt, kt)
        outputs.append(state @ qt)
    return np.array(outputs), state


def chunked(q, k, v, alpha, beta, initial, chunk):
    """A triangular linear system, independent of the token recurrence above."""
    state = initial.copy()
    outputs = []
    for start in range(0, len(q), chunk):
        sl = slice(start, start + chunk)
        qq, kk, vv, aa, bb = q[sl], k[sl], v[sl], alpha[sl], beta[sl]
        n = len(qq)
        # Block-local log-prefix differences avoid a long product over the sequence.
        log_gamma = np.cumsum(np.log(aa))
        gamma = np.exp(log_gamma)
        ratios = np.exp(log_gamma[:, None] - log_gamma[None, :])
        triangle = np.tril(bb[:, None] * ratios * (kk @ kk.T), -1)
        lower = np.eye(n) + triangle
        u = np.linalg.solve(lower, bb[:, None] * vv)
        w = np.linalg.solve(lower, (bb * gamma)[:, None] * kk)
        delta = u - w @ state.T
        within = np.tril((qq @ kk.T) * ratios)
        outputs.append(gamma[:, None] * (qq @ state.T) + within @ delta)
        state = gamma[-1] * state + delta.T @ (np.exp(log_gamma[-1] - log_gamma)[:, None] * kk)
    return np.concatenate(outputs), state


# A four-tap causal convolution. Coefficients run from oldest to newest.
TAPS = np.array([.1, .2, .3, .4])


def batch_conv(raw):
    """Explicit lag sum; no incremental convolution buffer is used."""
    result = np.zeros_like(raw)
    for t in range(len(raw)):
        for lag in range(4):
            if t >= lag:
                result[t] += TAPS[3-lag] * raw[t-lag]
    return result


def features(mixed, inputs):
    z = mixed / (1 + np.exp(-mixed))  # SiLU after the convolution.
    q, k, v = unit(z[:, :3]), unit(z[:, 3:6]), z[:, 6:8]
    alpha = np.exp(-.3 * np.logaddexp(0, inputs[:, 0]))
    beta = 1 / (1 + np.exp(-inputs[:, 1]))
    return q, k, v, alpha, beta


def full_reference(inputs, projection, chunk):
    raw = inputs @ projection
    q, k, v, a, b = features(batch_conv(raw), inputs)
    y, state = chunked(q, k, v, a, b, np.zeros((2, 3)), chunk)
    keys = y + .3 * inputs[:, :2]
    values = np.tanh(y + .2 * inputs[:, 2:])
    queries = y + .2 * inputs[:, :2]
    logits = queries @ keys.T / np.sqrt(2)
    logits[np.triu_indices(len(inputs), 1)] = -np.inf
    output = y + softmax(logits) @ values
    return output, state, raw[-3:].copy(), keys, values


@dataclass
class Cache:
    state: np.ndarray
    conv: np.ndarray
    keys: list
    values: list
    position: int = 0

    @classmethod
    def empty(cls):
        return cls(np.zeros((2, 3)), np.zeros((3, 8)), [], [])

    def clone(self):
        # A prefix checkpoint must own all mutable state, including convolution history.
        return Cache(self.state.copy(), self.conv.copy(),
                     [x.copy() for x in self.keys], [x.copy() for x in self.values], self.position)


def step(x, cache, projection):
    raw = x @ projection
    window = np.vstack([cache.conv, raw])
    mixed = TAPS @ window
    cache.conv = window[-3:].copy()
    q, k, v, a, b = features(mixed[None, :], x[None, :])
    cache.state *= a[0]
    cache.state += b[0] * np.outer(v[0] - cache.state @ k[0], k[0])
    y = cache.state @ q[0]
    cache.keys.append(y + .3*x[:2])
    cache.values.append(np.tanh(y + .2*x[2:]))
    cache.position += 1
    assert cache.position == len(cache.keys) == len(cache.values)
    query = y + .2*x[:2]
    scores = np.array(cache.keys) @ query / np.sqrt(2)
    return y + softmax(scores) @ np.array(cache.values)


def stream(inputs, cache, projection):
    return np.array([step(x, cache, projection) for x in inputs])


def prefill_block(inputs, cache, projection, chunk):
    """Block convolution, triangular GDN and causal attention over cached + new KV."""
    raw = inputs @ projection
    extended = np.concatenate([cache.conv, raw])
    mixed = batch_conv(extended)[len(cache.conv):]
    q, k, v, a, b = features(mixed, inputs)
    y, state = chunked(q, k, v, a, b, cache.state, chunk)
    new_keys = y + .3 * inputs[:, :2]
    new_values = np.tanh(y + .2 * inputs[:, 2:])
    old_length = cache.position
    all_keys = np.concatenate([np.array(cache.keys).reshape(-1, 2), new_keys])
    all_values = np.concatenate([np.array(cache.values).reshape(-1, 2), new_values])
    queries = y + .2 * inputs[:, :2]
    scores = queries @ all_keys.T / np.sqrt(2)
    future = np.arange(len(all_keys))[None, :] > old_length + np.arange(len(inputs))[:, None]
    scores[future] = -np.inf
    output = y + softmax(scores) @ all_values
    cache.state = state.copy()
    cache.conv = extended[-3:].copy()
    cache.keys.extend(row.copy() for row in new_keys)
    cache.values.extend(row.copy() for row in new_values)
    cache.position += len(inputs)
    assert cache.position == len(cache.keys) == len(cache.values)
    return output


def compare(actual, expected, title, atol=2e-12):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=2e-12)
    print('PASS', title, f'(max error {np.max(np.abs(actual-expected)):.3g})')


def scalar_rollback():
    """Hand-calculated case shared with the article's interactive figure."""
    def run(values, state=0., history=None):
        history = [0., 0., 0.] if history is None else list(history)
        for value in values:
            window = history + [float(value)]
            conv = sum(weight*x for weight, x in zip(TAPS, window))
            state = .5*state + .5*conv
            history = window[-3:]
        return state, history
    accepted, history = run([1, 2, 3, 4, 5])
    tentative, tentative_history = run([6], accepted, history)
    good, _ = run([7], accepted, history)
    bad, _ = run([7], tentative, tentative_history)
    state_only, _ = run([7], accepted, tentative_history)
    conv_only, _ = run([7], tentative, history)
    np.testing.assert_allclose([accepted, tentative, good, bad, state_only, conv_only],
                              [3.08125, 4.040625, 4.240625, 5.0203125, 4.540625, 4.7203125])
    print(f'PASS scalar rollback: restored {good:.7f}; KV-only {bad:.7f}; difference {bad-good:.7f}')
    print(f'  restoring S only leaves error {state_only-good:.7f}; conv only {conv_only-good:.7f}')


def worked_probabilities():
    # Exercise 1: independent hand-calculated scalar chunk.
    one = np.ones((2, 1))
    out, final = chunked(one, one, np.array([[2.], [4.]]),
                         np.full(2, .5), np.full(2, .5), np.zeros((1, 1)), 2)
    compare(out[:, 0], np.array([1., 2.25]), 'two-token scalar chunk')
    np.testing.assert_allclose(final, [[2.25]])
    # Exercise 5: each remote assignment dispatches one BF16 vector and returns one.
    remote_bytes = 256 * 10 * 2048 * 2 * 2 * (1 - 1/8)
    assert remote_bytes / 2**20 == 17.5
    assert remote_bytes * 48 / 2**20 == 840
    print('PASS EP8 assignment ledger: 17.5 MiB/layer, 840 MiB across 48 layers')
    # Exercise 6: enumerate the accept/correct law instead of sampling it.
    p = np.array([.5, .3, .2])
    q = np.array([.6, .1, .3])
    acceptance = np.minimum(1, p/q)
    accepted_mass = q * acceptance
    rejection = 1 - accepted_mass.sum()
    residual = np.maximum(p-q, 0)
    residual /= residual.sum()
    compare(acceptance, np.array([5/6, 1., 2/3]), 'speculative acceptance probabilities')
    np.testing.assert_allclose(rejection, .2)
    compare(residual, np.array([0., 1., 0.]), 'normalized correction distribution')
    compare(accepted_mass + rejection*residual, p, 'accept/correct reproduces target distribution')


def memory_counts(conv_slots):
    mib = 2**20
    kv_per_token = 12 * 2 * 2 * 256 * 2  # layers, K+V, KV heads, width, BF16 bytes
    state = 36 * 32 * 128 * 128 * 4      # GDN layers, V heads, key/value widths, FP32 bytes
    channels = 2 * 16 * 128 + 32 * 128  # convolution sees Q, K, V before Q/K head repetition
    conv = 36 * channels * conv_slots * 2
    assert kv_per_token == 24 * 1024 and state == 72 * mib
    assert 36 * channels * 3 * 2 == 1.6875 * mib
    assert 36 * channels * 4 * 2 == 2.25 * mib
    print(f'PASS memory: KV {kv_per_token//1024} KiB/token; FP32 recurrent state {state/mib:.0f} MiB')
    print(f'  BF16 convolution, {conv_slots} stored slots: {conv/mib:.4f} MiB per sequence')
    for context in (32768, 262144):
        total = kv_per_token * context + state + conv
        print(f'  context={context}: main-branch logical cache {total/mib:.4f} MiB/sequence')
    print('  Ledger includes main-branch KV, recurrent state and convolution buffer; add MTP, scales, padding and workspace by allocation.')


def expert_counts(rng, batch):
    e, k, tile = 512, 10, 16
    logits = rng.normal(size=(batch, e))
    # Continuous IID logits give a uniformly random k-subset per token.
    selected = np.argsort(-logits, axis=1, kind='stable')[:, :k]
    counts = np.bincount(selected.ravel(), minlength=e)
    assert counts.sum() == batch*k
    assert np.all([len(set(row)) == k for row in selected])
    probabilities = softmax(logits)
    p_top = np.take_along_axis(probabilities, selected, axis=1)
    normalized = p_top / p_top.sum(axis=1, keepdims=True)
    compare(normalized, softmax(np.take_along_axis(logits, selected, axis=1)), 'top-k renormalization')
    expected = e * (1 - (1-k/e)**batch)
    padded = tile * np.sum((counts + tile - 1)//tile)
    used = np.count_nonzero(counts)
    print(f'  batch={batch}: {used} routed experts used; uniform-routing expectation {expected:.6f}')
    print(f'  assignments={counts.sum()}, busiest expert={counts.max()}, padded rows={padded}, row utilization={counts.sum()/padded:.4%}')
    # Independently calculate marginal occupancy by enumerating Bernoulli events at small B.
    p = k/e
    for small_batch in (1, 2, 4, 8):
        occupied = 0.
        for pattern in range(1 << small_batch):
            n = pattern.bit_count()
            if n:
                occupied += p**n * (1-p)**(small_batch-n)
        np.testing.assert_allclose(occupied, 1-(1-p)**small_batch)
    # Identical token routes form a distinct workload: only k experts are touched.
    shared_counts = np.bincount(np.tile(np.arange(k), batch), minlength=e)
    assert np.count_nonzero(shared_counts) == k and shared_counts.sum() == batch*k
    print('PASS expected occupancy by enumeration; fully shared routes touch exactly 10 experts')
    # Selected routed weights sum to one. The shared expert has its own sigmoid gate.
    probs = np.array([.6, .3, .1])
    top_weights = probs[:2]/probs[:2].sum()
    shared_logit = np.log(.25/.75)
    shared_gate = 1/(1+np.exp(-shared_logit))
    output = top_weights @ np.array([1., 4.]) + shared_gate * 2
    np.testing.assert_allclose(output, 2.5)
    print('PASS routed output 2.0 + shared gate contribution 0.5 = 2.5')
    print('  Tile utilization is a row-count model, to be compared with real kernel profiles.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--chunk', type=int, default=4, choices=range(1, 33), metavar='1..32')
    parser.add_argument('--batch', type=int, default=32, choices=range(1, 513), metavar='1..512')
    parser.add_argument('--conv-slots', type=int, choices=(3, 4), default=3,
                        help='3 logical history positions, or a four-slot implementation buffer')
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    length, dk, dv = 19, 5, 3
    q, k = unit(rng.normal(size=(length, dk))), unit(rng.normal(size=(length, dk)))
    v = rng.normal(size=(length, dv))
    alpha, beta = rng.uniform(.45, .98, length), rng.uniform(.05, .95, length)
    initial = rng.normal(size=(dv, dk))
    reference, final = recurrent(q, k, v, alpha, beta, initial)
    for chunk in sorted({1, 2, 4, 5, 7, length, args.chunk}):
        out, state = chunked(q, k, v, alpha, beta, initial, chunk)
        compare(out, reference, f'GDN triangular blocks, chunk={chunk}')
        np.testing.assert_allclose(state, final, atol=2e-12)
    # A unit key and beta=1 exactly replace that association even after decay.
    key = unit(rng.normal(size=(1, dk)))[0]
    value = rng.normal(size=dv)
    one, _ = recurrent(key[None], key[None], value[None], np.array([.8]), np.ones(1), initial)
    compare(one[0], value, 'unit-key overwrite after decay')

    inputs = rng.normal(size=(20, 4))
    projection = rng.normal(size=(4, 8)) * .5
    full, final, history, keys, values = full_reference(inputs, projection, args.chunk)
    cache = Cache.empty()
    incremental = stream(inputs, cache, projection)
    compare(incremental, full, 'full causal prefill versus incremental hybrid decode')
    compare(cache.state, final, 'final recurrent state')
    compare(cache.conv, history, 'convolution history')
    segmented = Cache.empty()
    parts = [prefill_block(inputs[start:start+args.chunk], segmented, projection, args.chunk)
             for start in range(0, len(inputs), args.chunk)]
    compare(np.concatenate(parts), full, 'segmented triangular prefill with persistent state')
    compare(segmented.state, cache.state, 'segmented prefill final recurrent state')
    compare(np.array(segmented.keys), np.array(cache.keys), 'segmented prefill KV contents')
    # End the prefill at a partial block and continue with the token recurrence.
    boundary = 13
    hybrid = Cache.empty()
    parts = [prefill_block(inputs[start:min(start+args.chunk, boundary)], hybrid, projection, args.chunk)
             for start in range(0, boundary, args.chunk)]
    parts.append(stream(inputs[boundary:], hybrid, projection))
    compare(np.concatenate(parts), full, 'chunked prefill followed by token decode')
    compare(hybrid.state, cache.state, 'prefill/decode final recurrent state')
    compare(hybrid.conv, cache.conv, 'prefill/decode final convolution history')

    prefix = Cache.empty()
    stream(inputs[:5], prefix, projection)
    frozen = prefix.clone()
    left, right = prefix.clone(), prefix.clone()
    left_out = stream(inputs[5:8], left, projection)
    right_out = stream(inputs[12:15], right, projection)
    compare(left_out, full_reference(inputs[:8], projection, args.chunk)[0][5:], 'prefix fork A')
    other = np.concatenate([inputs[:5], inputs[12:15]])
    compare(right_out, full_reference(other, projection, args.chunk)[0][5:], 'prefix fork B')
    compare(prefix.state, frozen.state, 'shared prefix remains unchanged')
    compare(prefix.conv, frozen.conv, 'forks own their convolution history')
    assert prefix.position == 5 and not np.shares_memory(prefix.state, left.state)

    # Verify three tentative tokens; cover zero, one and all accepted.
    tentative = prefix.clone()
    stream(inputs[5:8], tentative, projection)
    for accepted in (0, 1, 3):
        restored = prefix.clone()
        stream(inputs[5:5+accepted], restored, projection)
        correct = step(inputs[12], restored, projection)
        accepted_sequence = np.concatenate([inputs[:5+accepted], inputs[12:13]])
        target = full_reference(accepted_sequence, projection, args.chunk)[0][-1]
        compare(correct, target, f'accepted={accepted}/3: restore/replay and continue')
        broken = tentative.clone()
        broken.keys = broken.keys[:5+accepted]
        broken.values = broken.values[:5+accepted]
        broken.position = 5+accepted
        wrong = step(inputs[12], broken, projection)
        if accepted < 3:
            difference = np.linalg.norm(wrong-target)
            assert difference > 1e-8
            print(f'PASS KV-only rollback counterexample, accepted={accepted}: output L2 error {difference:.6f}')
        else:
            compare(wrong, target, 'all accepted: tentative state is ready for continuation')
    scalar_rollback()
    worked_probabilities()
    memory_counts(args.conv_slots)
    expert_counts(rng, args.batch)
    print('All checks passed: chunk algebra, streaming state, prefix forks, rollback, memory, routing.')


if __name__ == '__main__':
    main()
