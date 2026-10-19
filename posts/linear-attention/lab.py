#!/usr/bin/env python3
"""线性 attention 的递推、dense、chunk 三条参考路径；CPU/NumPy。

运行: python3 lab.py --seed 7 --chunk 5 --length 19
phi 使用正值特征。改变为 ELU+1 时，三条路径必须一起替换。
所有路径采用 denominator + epsilon；这不是 softmax attention 的近似误差测试。
"""
import argparse
import numpy as np


def phi(x):
    # 正值映射，让分母有明确的非负权重含义。
    return np.maximum(x, 0) + 1


def recurrent(q, k, v, epsilon):
    state = np.zeros((q.shape[-1], v.shape[-1]), dtype=np.float64)
    z = np.zeros(q.shape[-1], dtype=np.float64)
    outputs = []
    for qi, ki, vi in zip(q, k, v):
        state += np.outer(ki, vi)
        z += ki
        outputs.append((qi @ state) / (qi @ z + epsilon))
    return np.asarray(outputs)


def dense_kernel(q, k, v, epsilon):
    pair = (q @ k.T) * np.tril(np.ones((len(q), len(q))))
    return (pair @ v) / (pair.sum(axis=-1, keepdims=True) + epsilon)


def chunkwise(q, k, v, epsilon, size):
    state = np.zeros((q.shape[-1], v.shape[-1]), dtype=np.float64)
    z = np.zeros(q.shape[-1], dtype=np.float64)
    outputs = []
    for start in range(0, len(q), size):
        qc, kc, vc = q[start:start+size], k[start:start+size], v[start:start+size]
        pair = (qc @ kc.T) * np.tril(np.ones((len(qc), len(qc))))
        numerator = qc @ state + pair @ vc
        denominator = qc @ z + pair.sum(axis=-1)
        outputs.append(numerator / (denominator[:, None] + epsilon))
        state += kc.T @ vc
        z += kc.sum(axis=0)
    return np.concatenate(outputs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--chunk', type=int, default=4)
    ap.add_argument('--length', type=int, default=19)
    args = ap.parse_args()
    if min(args.chunk, args.length) < 1:
        ap.error('--chunk 和 --length 必须为正整数')
    rng = np.random.default_rng(args.seed)
    q, k = phi(rng.normal(size=(2, args.length, 6)))
    v = rng.normal(size=(args.length, 4))
    epsilon = 1e-8
    ref = recurrent(q, k, v, epsilon)
    np.testing.assert_allclose(ref, dense_kernel(q, k, v, epsilon), atol=1e-12)
    print('recurrent vs dense kernel: PASS')
    np.testing.assert_allclose(ref, chunkwise(q, k, v, epsilon, args.chunk), atol=1e-12)
    print('recurrent vs chunkwise: PASS')
    # 手算二维 oracle，独立于随机路径对照。
    kt = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    vt = np.array([[2, 1], [4, 3], [3, 2]], dtype=float)
    qt = np.array([[1, 1], [1, 1], [2, 1]], dtype=float)
    np.testing.assert_allclose(recurrent(qt, kt, vt, 0)[-1], [17/6, 11/6])
    # 标量门控递推，对照直接构造每个历史位置的门乘积。
    values = np.array([2., 4., 8.])
    alpha = np.array([1., .5, .25])
    state = z = 0.
    for a, value in zip(alpha, values):
        state, z = a * state + value, a * z + 1
    coeff = np.array([np.prod(alpha[i+1:]) for i in range(len(alpha))])
    np.testing.assert_allclose(state / z, coeff @ values / coeff.sum())
    np.testing.assert_allclose(state / z, 74/11)
    print('gated scan coefficients: PASS')
    # 单位 key 的误差应每次缩小一半；从 S=2 逼近目标 5。
    state = 2.
    for _ in range(3):
        state += .5 * (5 - state)
    np.testing.assert_allclose(state, 4.625)
    print('delta unit-key convergence: PASS')
    # 非单位 key 的过冲反例。
    nonunit = 1 + .5 * 2 * (5 - 2 * 1)
    assert 2 * nonunit == 8
    assert (128 * 128 + 128) * 4 / (2 * 128 * 2) == 129


if __name__ == '__main__':
    main()
