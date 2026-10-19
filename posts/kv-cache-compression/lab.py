#!/usr/bin/env python3
"""KV 量化误差与预算的 CPU/NumPy 教学实验。

运行: python3 lab.py --seed 7 --group 32
整数码为了可读性存为 int8；下方 INT4 字节是打包格式预算，不是 q.nbytes。
对称量化只存 scale；元数据练习单独使用 scale + zero 的仿射存储假设。
"""
import argparse
import numpy as np


def quantize_groups(x, group, axis=-1):
    a = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    rebuilt = np.empty_like(a)
    codes = np.empty_like(a, dtype=np.int8)
    bounds = np.empty_like(a)
    for start in range(0, a.shape[-1], group):
        block = a[..., start:start + group]
        scale = np.max(np.abs(block), axis=-1, keepdims=True) / 7
        scale = np.where(scale == 0, 1, scale)
        q = np.clip(np.rint(block / scale), -7, 7).astype(np.int8)
        codes[..., start:start + group] = q
        rebuilt[..., start:start + group] = q * scale
        bounds[..., start:start + group] = scale / 2
    return (np.moveaxis(rebuilt, -1, axis), np.moveaxis(codes, -1, axis),
            np.moveaxis(bounds, -1, axis))


def attn(q, k, v):
    score = q @ k.T / np.sqrt(k.shape[-1])
    score -= score.max(axis=-1, keepdims=True)
    a = np.exp(score)
    a /= a.sum(axis=-1, keepdims=True)
    return a @ v


def affine_int4_bytes(elements, group):
    # 四舍五入到完整 byte，以及完整的最后一组元数据。
    return (elements + 1) // 2 + ((elements + group - 1) // group) * 4


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--group', type=int, default=64)
    args = ap.parse_args()
    if args.group < 1:
        ap.error('--group 必须为正整数')
    toy = np.array([-1., -.2, .3, 1.])
    rebuilt, q, bound = quantize_groups(toy, 4)
    np.testing.assert_array_equal(q, [-7, -1, 2, 7])
    np.testing.assert_allclose(rebuilt, [-1, -1/7, 2/7, 1])
    np.testing.assert_allclose(np.max(abs(toy - rebuilt)), 2/35)
    assert np.all(abs(toy - rebuilt) <= bound + 1e-12)
    rng = np.random.default_rng(args.seed)
    k = rng.normal(size=(128, 64))
    v = rng.normal(size=(128, 32))
    query = rng.normal(size=(4, 64))
    kq, _, kb = quantize_groups(k, args.group, axis=0)
    vq, _, vb = quantize_groups(v, args.group, axis=1)
    assert np.all(abs(k - kq) <= kb + 1e-12)
    assert np.all(abs(v - vq) <= vb + 1e-12)
    ref = attn(query, k, v)
    print('INT4 round-trip bound: PASS')
    print('K-only / V-only error measured')
    print(f'  max output error K: {np.max(abs(attn(query,kq,v)-ref)):.6f}')
    print(f'  max output error V: {np.max(abs(attn(query,k,vq)-ref)):.6f}')
    n = 1048576
    total = affine_int4_bytes(n, 64)
    assert total == 589824
    print(f'metadata: {total / 1024:.0f} KiB; ratio = {2*n/total:.3f}')
    # 旧 token 按 64 分组：每组 4 byte scale/zero；近期 128 token 保留 BF16。
    old_elements = 2 * (8192 - 128) * 8 * 128
    residual = 2 * 128 * 8 * 128 * 2
    all_layers = 32 * (affine_int4_bytes(old_elements, 64) + residual)
    assert all_layers == int(299.5 * 2**20)
    print(f'residual example: {all_layers / 2**20:.3f} MiB')
    # 相同 outlier 数据，比较分组轴；这不是完整 KIVI 实现。
    k[:, 0] *= 20
    token_rebuilt, _, _ = quantize_groups(k, args.group, axis=1)
    channel_rebuilt, _, _ = quantize_groups(k, args.group, axis=0)
    # 分别报告，避免预先假定任意输入的优劣顺序。
    print(f'outlier K MSE, token groups: {np.mean((k-token_rebuilt)**2):.6f}')
    print(f'outlier K MSE, channel groups: {np.mean((k-channel_rebuilt)**2):.6f}')


if __name__ == '__main__':
    main()
