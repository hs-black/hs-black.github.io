#!/usr/bin/env python3
"""单层 GQA + RoPE 教学参考；CPU/NumPy，不是性能 benchmark。

运行: python3 lab.py --seed 7 --chunk 3
修改方向: 把 q_heads 从 4 改成 8，保持 kv_heads=2；检查 cache 不变。
故意将 incremental() 的 position offset 或 causal offset 改成 0，断言应失败。
"""
import argparse
import numpy as np


def rope(x, positions):
    """x: [token, head, d]；将相邻两维作为旋转对。"""
    d = x.shape[-1]
    angle = np.asarray(positions)[:, None, None] / 10000.0 ** (
        np.arange(0, d, 2)[None, None, :] / d
    )
    c, s = np.cos(angle), np.sin(angle)
    out = np.empty_like(x)
    out[..., 0::2] = x[..., 0::2] * c - x[..., 1::2] * s
    out[..., 1::2] = x[..., 0::2] * s + x[..., 1::2] * c
    return out


def attention(q, k, v, allowed):
    # repeat 只为参考实现清楚；生产 kernel 应直接复用 KV。
    repeats = q.shape[1] // k.shape[1]
    kh = np.repeat(k, repeats, axis=1)
    vh = np.repeat(v, repeats, axis=1)
    scores = np.einsum('thd,shd->hts', q, kh) / np.sqrt(q.shape[-1])
    scores = np.where(allowed[None, :, :], scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum('hts,shd->thd', weights, vh)


def project(x, wq, wk, wv, q_heads, kv_heads, d):
    return ((x @ wq).reshape(-1, q_heads, d),
            (x @ wk).reshape(-1, kv_heads, d),
            (x @ wv).reshape(-1, kv_heads, d))


def incremental(x, weights, heads, d, chunks, wrong_mask=False):
    qh, kh = heads
    kc = np.empty((0, kh, d))
    vc = np.empty_like(kc)
    outputs = []
    start = 0
    for size in chunks:
        end = min(start + size, len(x))
        if start == end:
            break
        q, k, v = project(x[start:end], *weights, qh, kh, d)
        positions = np.arange(start, end)
        q, k = rope(q, positions), rope(k, positions)
        kc, vc = np.concatenate((kc, k)), np.concatenate((vc, v))
        # P + i，绝对位置；左上对齐的无偏移三角在 chunk decode 会错。
        query_positions = np.arange(end - start) if wrong_mask else positions
        allowed = np.arange(end)[None, :] <= query_positions[:, None]
        outputs.append(attention(q, kc, vc, allowed))
        start = end
    assert start == len(x), 'chunks 必须覆盖全部输入'
    return np.concatenate(outputs), kc, vc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--chunk', type=int, default=3)
    args = ap.parse_args()
    if args.chunk < 1:
        ap.error('--chunk 必须为正整数')
    rng = np.random.default_rng(args.seed)
    n, model_dim, qh, kh, d = 11, 12, 4, 2, 4
    x = rng.normal(size=(n, model_dim))
    weights = [rng.normal(size=(model_dim, h * d)) / np.sqrt(model_dim)
               for h in (qh, kh, kh)]
    q, k, v = project(x, *weights, qh, kh, d)
    q, k = rope(q, np.arange(n)), rope(k, np.arange(n))
    dense = attention(q, k, v, np.tril(np.ones((n, n), dtype=bool)))
    by_token, kc, vc = incremental(x, weights, (qh, kh), d, [1] * n)
    by_chunk, _, _ = incremental(x, weights, (qh, kh), d, [args.chunk] * n)
    np.testing.assert_allclose(dense, by_token, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(dense, by_chunk, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(kc, k, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(vc, v, atol=1e-12, rtol=1e-12)
    wrong, _, _ = incremental(x, weights, (qh, kh), d, [1] * n, wrong_mask=True)
    assert np.max(np.abs(wrong - dense)) > 1e-3
    assert kc.size + vc.size == 2 * n * kh * d
    print('full vs token: PASS')
    print('full vs chunks: PASS')
    print('wrong mask detected: PASS')
    per_token = 2 * 32 * 8 * 128 * 2
    print(f'{per_token / 1024:.0f} KiB/token; 8192 tokens = {per_token * 8192 / 2**30:.3f} GiB')
    # 独立核对课后题：8192 输入 + 2048 输出的可保证并发。
    assert (16 * 2**30) // (per_token * 10240) == 12


if __name__ == '__main__':
    main()
