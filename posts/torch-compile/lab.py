#!/usr/bin/env python3
"""A small, explicit guard-cache model. This is NOT a PyTorch compiler.

python3 lab.py [--lengths 64,96,128,96,64,192 --compile-ms 600]
Strategies: exact static shapes, symbolic length with branch guards, buckets.
Cost figures are inputs in a teaching model; no actual kernel is benchmarked.
"""
from dataclasses import dataclass
import argparse
import math
import numpy as np


@dataclass(frozen=True)
class Input:
    length: int
    dtype: str = 'float32'
    contiguous: bool = True
    scale: float = 1.0


def branch(length):
    return 'short' if length <= 128 else 'long'


def guard_key(x, strategy):
    if strategy == 'static':
        shape = x.length
    elif strategy == 'symbolic':
        shape = branch(x.length)   # branch's length <= 128 remains a guard
    elif strategy == 'bucket':
        shape = 128 if x.length <= 128 else 256
    else:
        raise ValueError(strategy)
    # Tensor layout/dtype and a Python scalar can still specialize a graph.
    return shape, x.dtype, x.contiguous, x.scale


def reference(x):
    """Independent eager expression; batch axis is the variable length."""
    y = np.sin(x) + 2.0
    z = y * y
    return z if len(x) <= 128 else z / 2.0


def fused_scalar_reference(x):
    # Scalar loop provides a second calculation path, not the same array AST.
    factor = 1.0 if len(x) <= 128 else 0.5
    return np.array([factor * (math.sin(float(v)) + 2.0) ** 2 for v in x])


def costs(inputs, strategy, compile_ms=600.0):
    seen, rows = set(), []
    total = 0.0
    for x in inputs:
        key = guard_key(x, strategy)
        miss = key not in seen
        seen.add(key)
        extent = (128 if x.length <= 128 else 256) if strategy == 'bucket' else x.length
        # Simple linear elementwise cost, NOT a performance claim.
        run_ms = 0.05 * extent
        elapsed = run_ms + (compile_ms if miss else 0.0)
        total += elapsed
        rows.append((x.length, extent, miss, elapsed))
    return len(seen), total, rows


def self_test():
    xs = [Input(n) for n in (64, 96, 128, 96, 64, 192)]
    assert costs(xs, 'static')[0] == 4
    assert costs(xs, 'symbolic')[0] == 2
    assert costs(xs, 'bucket')[0] == 2
    assert costs([Input(64), Input(64, scale=2)], 'symbolic')[0] == 2
    assert costs([Input(64), Input(64, contiguous=False)], 'symbolic')[0] == 2
    assert costs([Input(64), Input(64, dtype='float64')], 'symbolic')[0] == 2
    assert guard_key(Input(128), 'symbolic') != guard_key(Input(129), 'symbolic')
    for n in (1, 64, 128, 129, 192, 256):
        x = np.linspace(-3, 3, n, dtype=np.float64)
        np.testing.assert_allclose(reference(x), fused_scalar_reference(x), atol=1e-12, rtol=1e-12)
        bucket = 128 if n <= 128 else 256
        padded = np.pad(x, (0, bucket-n))
        # Branch boundaries coincide with bucket boundaries here. Cropping
        # preserves outputs for this independent elementwise function.
        np.testing.assert_allclose(reference(padded)[:n], reference(x), atol=1e-12)
    # Counterexample: padding without a reduction mask changes mean.
    x = np.array([1., 2., 3.])
    assert x.mean() != np.pad(x, (0, 1)).mean()
    assert math.ceil(600/(8-5)) == 200


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lengths', default='64,96,128,96,64,192')
    p.add_argument('--compile-ms', type=float, default=600)
    a = p.parse_args()
    try:
        lengths = [int(n) for n in a.lengths.split(',')]
    except ValueError:
        p.error('lengths must be comma-separated integers')
    if not lengths or min(lengths) < 1 or max(lengths) > 256 or a.compile_ms < 0:
        p.error('lengths must be 1..256; compile-ms must be nonnegative')
    self_test()
    print('Teaching guard-cache model; no torch.compile/GPU timings are measured.')
    for strategy in ('static', 'symbolic', 'bucket'):
        count, total, rows = costs([Input(n) for n in lengths], strategy, a.compile_ms)
        print(f'{strategy}: variants={count} total={total:.3f} ms')
        print('  ' + ' | '.join(f'{n}->{extent}:{"compile" if miss else "hit"}' for n, extent, miss, _ in rows))
    threshold = a.compile_ms / (8 - 5)
    print(f'break-even (eager 8 ms, compiled 5 ms): N > {threshold:g}; first strictly faster integer N={math.floor(threshold)+1}')
    print('PASS: cache keys, branch boundary, dtype/layout/scalar guards, numerics, padding counterexample')


if __name__ == '__main__':
    main()
