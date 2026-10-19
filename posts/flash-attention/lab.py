#!/usr/bin/env python3
"""FlashAttention 的 CPU 数学实验（NumPy；不是 GPU 性能实现）。

python3 lab.py --length 7 --block 3
改变 block 不应改变数学结果；修改 rescale 或 causal mask 应触发断言。
前向用逐行 online softmax，反向分块恢复概率；独立对照为完整矩阵
softmax 和损失的中心差分。所有测试使用 float64 隔离量化误差。
"""
import argparse
import numpy as np


def dense(q, k, v, causal=False):
    scores = q @ k.T / np.sqrt(q.shape[1])
    if causal:
        scores = np.where(np.arange(len(k))[None, :] <= np.arange(len(q))[:, None], scores, -np.inf)
    maximum = scores.max(axis=1, keepdims=True)
    e = np.exp(scores - maximum)
    p = e / e.sum(axis=1, keepdims=True)
    return p @ v


def tiled(q, k, v, block, causal=False):
    """每次只保留一行和一个 key tile，返回 O 和每行 logsumexp。"""
    out = np.zeros((len(q), v.shape[1]))
    lse = np.empty(len(q))
    for i, qi in enumerate(q):
        m, denominator = -np.inf, 0.0
        numerator = np.zeros(v.shape[1])
        for start in range(0, len(k), block):
            stop = min(start + block, len(k), i + 1 if causal else len(k))
            if stop <= start:  # 整块被 causal mask 遮住，跳过。
                continue
            scores = qi @ k[start:stop].T / np.sqrt(q.shape[1])
            new_m = max(m, scores.max())
            alpha = np.exp(m - new_m) if denominator else 0.0
            e = np.exp(scores - new_m)
            denominator = alpha * denominator + e.sum()
            numerator = alpha * numerator + e @ v[start:stop]
            m = new_m
        out[i] = numerator / denominator
        lse[i] = m + np.log(denominator)
    return out, lse


def backward(q, k, v, out, lse, dout, block, causal=False):
    dq, dk, dv = np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)
    scale = 1 / np.sqrt(q.shape[1])
    row_dot = np.sum(dout * out, axis=1)
    for i in range(len(q)):
        for start in range(0, len(k), block):
            stop = min(start + block, len(k), i + 1 if causal else len(k))
            if stop <= start:
                continue
            p = np.exp((q[i] @ k[start:stop].T) * scale - lse[i])
            dp = dout[i] @ v[start:stop].T
            ds = p * (dp - row_dot[i])
            dq[i] += ds @ k[start:stop] * scale
            dk[start:stop] += ds[:, None] * q[i] * scale
            dv[start:stop] += p[:, None] * dout[i]
    return dq, dk, dv


def finite_difference(q, k, v, dout, causal):
    grads = []
    for array in (q, k, v):
        grad = np.zeros_like(array)
        for index in np.ndindex(array.shape):
            old, h = array[index], 1e-5
            array[index] = old + h
            plus = np.sum(dense(q, k, v, causal) * dout)
            array[index] = old - h
            minus = np.sum(dense(q, k, v, causal) * dout)
            array[index] = old
            grad[index] = (plus - minus) / (2 * h)
        grads.append(grad)
    return grads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length', type=int, default=7)
    parser.add_argument('--block', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.length <= 64 or args.block < 1:
        parser.error('length 应在 1..64，block 应 >= 1')
    rng = np.random.default_rng(7)
    q, k, v, do = [rng.normal(size=(args.length, 3)) for _ in range(4)]
    for causal in (False, True):
        out, lse = tiled(q, k, v, args.block, causal)
        ref = dense(q, k, v, causal)
        error = np.max(np.abs(out - ref))
        np.testing.assert_allclose(out, ref, atol=1e-12, rtol=1e-12)
        analytical = backward(q, k, v, out, lse, do, args.block, causal)
        numerical = finite_difference(q, k, v, do, causal)
        grad_error = max(np.max(np.abs(a - b)) for a, b in zip(analytical, numerical))
        for a, b in zip(analytical, numerical):
            np.testing.assert_allclose(a, b, atol=2e-8, rtol=2e-6)
        print(f'causal={causal}: forward error={error:.2e}, gradient error={grad_error:.2e}')
    # 手算的独立常量：相同分数应给出 value 的平均数。
    equal, _ = tiled(np.zeros((1, 2)), np.zeros((3, 2)), np.array([[1.], [4.], [7.]]), args.block)
    np.testing.assert_allclose(equal, [[4.]])
    # 大分数仍有限；稳定减最大值要在指数之前。
    large, _ = tiled(q * 100, k * 100, v, args.block)
    assert np.isfinite(large).all()
    print('PASS: dense parity, Q/K/V finite differences, uniform scores, large scores')


if __name__ == '__main__':
    main()
