#!/usr/bin/env python3
"""GEMM 数值与流量实验。python3 lab.py --tile 4 4 4

依赖 NumPy。用显式三重循环和 NumPy matmul 双重检查 blocked GEMM。
流量模型假设每个输出 tile 自行加载 A/B、没有跨 block 的 cache 复用；
只算有效元素，不把 masked padding 算作真实 HBM 读。它不是 GPU 实测。
"""
import argparse
import math
import numpy as np


def scalar_reference(a, b):
    m, k = a.shape
    n = b.shape[1]
    out = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            for q in range(k):
                out[i, j] += float(a[i, q]) * float(b[q, j])
    return out


def tiled(a, b, bm, bn, bk):
    m, k = a.shape
    n = b.shape[1]
    c = np.zeros((m, n), dtype=np.float64)
    loads = stores = padded_fma = 0
    for i in range(0, m, bm):
        for j in range(0, n, bn):
            # 固定大小 tile；尾块填零，相当于 masked tl.load(..., other=0)。
            acc = np.zeros((bm, bn), dtype=np.float64)
            im, jn = min(bm, m-i), min(bn, n-j)
            for q in range(0, k, bk):
                qk = min(bk, k-q)
                at = np.zeros((bm, bk))
                bt = np.zeros((bk, bn))
                at[:im, :qk] = a[i:i+im, q:q+qk]
                bt[:qk, :jn] = b[q:q+qk, j:j+jn]
                # 用 outer product 累加，独立于参考的 i,j,k 路径。
                for r in range(bk):
                    acc += at[:, r:r+1] * bt[r:r+1, :]
                loads += im*qk + qk*jn
                padded_fma += bm*bn*bk
            c[i:i+im, j:j+jn] = acc[:im, :jn]
            stores += im*jn
    return c, loads, stores, padded_fma


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tile', type=int, nargs=3, default=[4, 4, 4], metavar=('BM', 'BN', 'BK'))
    args = parser.parse_args()
    if any(x <= 0 for x in args.tile):
        parser.error('tile sizes must be positive')
    bm, bn, bk = args.tile
    rng = np.random.default_rng(7)
    for m, n, k in [(8, 8, 8), (7, 9, 5), (1, 13, 7), (17, 3, 1)]:
        a = rng.normal(size=(m, k))
        b = rng.normal(size=(k, n))
        got, loads, stores, fma = tiled(a, b, bm, bn, bk)
        np.testing.assert_allclose(got, scalar_reference(a, b), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(got, a @ b, atol=1e-12, rtol=1e-12)
        # 独立推导：每列输出 tile 重读一次 A，每行 tile 重读一次 B。
        expected_loads = m*k*math.ceil(n/bn) + k*n*math.ceil(m/bm)
        expected_fma = math.ceil(m/bm)*math.ceil(n/bn)*math.ceil(k/bk)*bm*bn*bk
        assert loads == expected_loads and stores == m*n and fma == expected_fma
        print(f'M,N,K={m},{n},{k}: PASS; loads={loads}, stores={stores}, useful/padded FMA={m*n*k}/{fma}')
    bm, bn, bk, elem, stages = 64, 64, 32, 2, 2
    smem = stages * (bm*bk + bk*bn) * elem
    assert smem == 16*2**10
    print(f'64x64x32: double-buffer operands={smem // 1024} KiB; FP32 accumulator={bm*bn*4//1024} KiB')
    m = n = k = 4096
    flop = 2*m*n*k
    lower_bytes = 2*(m*k+k*n+m*n)
    assert lower_bytes == 96*2**20
    print(f'4096 cube: {flop/1e9:.3f} GFLOP, one-read/write lower bound={lower_bytes//2**20} MiB, AI={flop/lower_bytes:.2f} FLOP/byte')
    # 概念 tile 的单 stage 输入强度，不包括 C 写回。
    stage_ai = (2*64*64*32) / ((64*32+32*64)*2)
    assert stage_ai == 32
    print(f'64x64x32 stage operand AI={stage_ai:.0f} FLOP/byte')
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
