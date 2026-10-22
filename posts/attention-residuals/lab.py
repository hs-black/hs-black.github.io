#!/usr/bin/env python3
"""CPU / NumPy laboratory for Attention Residuals.

Run: python lab.py
Checks gradients against finite differences, exact grouped softmax, source-bank
boundaries, zero-query learning, and the article's byte counts. No model download.
All arithmetic here uses float64; timings are not GPU performance predictions.
"""
from __future__ import annotations

import math
import numpy as np


def rms_keys(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float64)
    return values / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + eps)


def softmax(scores):
    exp = np.exp(scores - np.max(scores))
    return exp / exp.sum()


def attend(query, values, eps=1e-8):
    keys = rms_keys(values, eps)
    weights = softmax(keys @ query)
    return weights @ values, weights


def query_gradient(query, values, upstream, eps=1e-8):
    output, weights = attend(query, values, eps)
    return rms_keys(values, eps).T @ (weights * ((values - output) @ upstream))


def value_gradient(query, values, upstream, eps=1e-8):
    output, weights = attend(query, values, eps)
    rms = np.sqrt(np.mean(values * values, axis=-1) + eps)
    # Jacobian-vector product of RMSNorm; gamma is fixed to 1 in this lab.
    route = query[None, :] / rms[:, None]
    route -= values * (values @ query)[:, None] / (values.shape[1] * rms[:, None] ** 3)
    score_grad = weights * ((values - output) @ upstream)
    return weights[:, None] * upstream + score_grad[:, None] * route


def finite_difference(function, value, step=1e-5):
    numerical = np.zeros_like(value)
    for index in np.ndindex(value.shape):
        left, right = value.copy(), value.copy()
        left[index] -= step
        right[index] += step
        numerical[index] = (function(right) - function(left)) / (2 * step)
    return numerical


def statistics(scores, values):
    maximum = float(np.max(scores))
    shifted = np.exp(scores - maximum)
    return maximum, float(shifted.sum()), shifted @ values


def merge(left, right):
    ml, zl, ul = left
    mr, zr, ur = right
    maximum = max(ml, mr)
    cl, cr = np.exp(ml - maximum), np.exp(mr - maximum)
    return maximum, cl * zl + cr * zr, cl * ul + cr * ur


def source_trace(embedding, outputs, block_size):
    """Banks immediately before each sublayer, followed by final output bank."""
    completed, partial = [embedding.copy()], None
    trace = []
    for index, value in enumerate(outputs):
        trace.append(np.stack(completed + ([] if partial is None else [partial])))
        partial = value.copy() if partial is None else partial + value
        if (index + 1) % block_size == 0 or index + 1 == len(outputs):
            completed.append(partial)
            partial = None
    trace.append(np.stack(completed))
    return trace


def main():
    rng = np.random.default_rng(7)
    values, query, upstream = rng.normal(size=(5, 4)), rng.normal(size=4), rng.normal(size=4)
    analytic_q = query_gradient(query, values, upstream)
    numeric_q = finite_difference(lambda q: attend(q, values)[0] @ upstream, query)
    analytic_v = value_gradient(query, values, upstream)
    numeric_v = finite_difference(lambda v: attend(query, v)[0] @ upstream, values)
    np.testing.assert_allclose(analytic_q, numeric_q, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(analytic_v, numeric_v, rtol=2e-7, atol=2e-8)
    print('PASS query gradient; max error:', np.max(np.abs(analytic_q - numeric_q)))
    print('PASS value gradient, including routing; max error:', np.max(np.abs(analytic_v - numeric_v)))

    # Independent invariants: constant values give exactly that output and zero q gradient.
    repeated = np.repeat([[1., -2., 3., .5]], 5, axis=0)
    np.testing.assert_allclose(attend(query, repeated)[0], repeated[0], atol=1e-14)
    np.testing.assert_allclose(query_gradient(query, repeated, upstream), 0, atol=1e-14)
    print('PASS identical-source invariant')

    # Compare arbitrary grouping against one full softmax, including large logits.
    maximum_error = 0.
    for offset in (0., 1000., -1000.):
        scores = rng.normal(size=17) * 20 + offset
        vals = rng.normal(size=(17, 4))
        reference = softmax(scores) @ vals
        for cuts in ([1, 6, 11], [4, 5, 16], [8]):
            groups = np.split(np.arange(len(scores)), cuts)
            result = statistics(scores[groups[0]], vals[groups[0]])
            for group in groups[1:]:
                result = merge(result, statistics(scores[group], vals[group]))
            _, denominator, numerator = result
            actual = numerator / denominator
            maximum_error = max(maximum_error, np.max(np.abs(actual - reference)))
            np.testing.assert_allclose(actual, reference, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(softmax(scores), softmax(scores + 1000), atol=1e-12)
    print('PASS online softmax across groups and extreme scores; max error:', maximum_error)

    # Block boundaries are checked against explicit hand-calculated source banks.
    embedding = np.array([10.])
    outputs = np.arange(1., 7.)[:, None]
    expected = [[10.], [10., 1.], [10., 3.], [10., 3., 3.],
                [10., 3., 7.], [10., 3., 7., 5.], [10., 3., 7., 11.]]
    trace = source_trace(embedding, outputs, 2)
    for actual, wanted in zip(trace, expected):
        np.testing.assert_array_equal(actual[:, 0], wanted)
    # Block size 1 preserves each raw output individually, including the embedding.
    for index, actual in enumerate(source_trace(embedding, outputs, 1)):
        np.testing.assert_array_equal(actual, np.vstack([embedding[None], outputs[:index]]))
    # An incomplete final block is flushed as a completed block.
    np.testing.assert_array_equal(source_trace(embedding, outputs[:5], 2)[-1][:, 0], [10, 3, 7, 5])
    # Empty partial state must not introduce an extra source into softmax.
    output, _ = attend(np.zeros(1), np.array([[10.]]))
    padded, _ = attend(np.zeros(1), np.array([[10.], [0.]]))
    assert output[0] == 10 and padded[0] == 5
    print('PASS block boundaries, Full granularity, incomplete block, empty-source counterexample')

    toy_values = np.array([[1., 0.], [0., 2.], [-1., 0.]])
    target = np.array([0., 2.])
    toy_query = np.zeros(2)
    first_output, _ = attend(toy_query, toy_values)
    initial_loss = .5 * np.sum((first_output - target) ** 2)
    first_gradient = query_gradient(toy_query, toy_values, first_output - target)
    assert np.linalg.norm(first_gradient) > .1
    for _ in range(400):
        output, _ = attend(toy_query, toy_values)
        toy_query -= .15 * query_gradient(toy_query, toy_values, output - target)
    final_output, final_weights = attend(toy_query, toy_values)
    final_loss = .5 * np.sum((final_output - target) ** 2)
    assert final_loss < initial_loss / 100
    print(f'PASS zero-initialized query learns: loss {initial_loss:.6f} -> {final_loss:.6f}')
    print('  query:', toy_query.round(4), 'weights:', final_weights.round(4))

    # Closed-form checks for article exercises; MiB=2**20 bytes.
    q_example = np.array([1., 0.])
    output, weights = attend(q_example, toy_values, eps=0)
    print('Exercise 1 weights:', weights.round(6), 'output:', output.round(6))
    exercise2 = query_gradient(np.zeros(2), np.eye(2), np.array([1., 0.]), eps=0)
    np.testing.assert_allclose(exercise2, np.sqrt(2) / 4 * np.array([1., -1.]))
    sublayers, block_size, batch, tokens, width, bytes_per_element = 96, 12, 2, 16384, 4096, 2
    sources = math.ceil(sublayers / block_size) + 1
    bank_bytes = sources * batch * tokens * width * bytes_per_element
    assert bank_bytes / 2**20 == 2304
    maximum_tokens = (128 * 2**20 * 8) // (sources * batch * width * bytes_per_element)
    assert maximum_tokens == 7281
    direct = softmax(np.array([0., 1., 2.])) @ np.array([2., -1., 3.])
    print(f'Exercise 3 source bank: {bank_bytes / 2**20:.0f} MiB; /8 = {bank_bytes / 8 / 2**20:.0f} MiB; chunk <= {maximum_tokens}')
    print(f'Exercise 4 correct merged output: {direct:.6f}')
    print('All numerical checks passed.')


if __name__ == '__main__':
    main()
