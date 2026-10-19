#!/usr/bin/env python3
"""V4 的容量与数据流 CPU 教学实验（仅标准库）。
python3 lab.py --context 32768 --batch 8 --capacity 80
context 表示参考实现的分配长度上限；容量单位 GiB。
练习：增大 context/batch，预测缓存增幅；改通信精度，比较 payload 与启动延迟。
"""
import argparse
import math

MODEL = 'pro'
CONFIG = {
    'flash': dict(d=4096, m=2048, layers=43, experts=256, groups=8, heads=64, tp=4),
    'pro': dict(d=7168, m=3072, layers=61, experts=384, groups=16, heads=128, tp=8),
}[MODEL]


def ratios():
    opening = [128, 128] if MODEL == 'pro' else [0, 0]
    return opening + [4 if i % 2 == 0 else 128 for i in range(2, CONFIG['layers'])]


def cache_bytes(context):
    """主干；BF16 cache，FP32 compressor 状态。未计 MTP、RoPE 表、workspace。"""
    main = index = state = 0
    for ratio in ratios():
        main += (128 + (context // ratio if ratio else 0)) * 512 * 2
        if ratio == 4:
            index += (context // 4) * 128 * 2
            state += 2 * 8 * 1024 * 4 + 2 * 8 * 256 * 4
        elif ratio == 128:
            state += 2 * 128 * 512 * 4
    return main, index, state


def routed_storage():
    # 按三块矩阵分别数 packed values 和 8-bit scale，而非使用总参数估计。
    d, m = CONFIG['d'], CONFIG['m']
    per_expert = 0
    for rows, cols in [(m, d), (d, m), (m, d)]:
        assert cols % 32 == 0
        per_expert += rows * (cols // 2) + rows * (cols // 32)
    return per_expert * CONFIG['layers'] * CONFIG['experts']


def valid_tp():
    return [g for g in range(1, CONFIG['groups']+1)
            if all(size % g == 0 for size in [CONFIG['heads'], CONFIG['groups'], 64, CONFIG['experts']])]


def ring_all_reduce(rows):
    """每 rank 有 G 个标量块；显式 reduce-scatter + all-gather。
    教学模拟只验证数据流，不模拟网络并发和真实 GPU 延迟。
    """
    g = len(rows)
    x = [list(row) for row in rows]
    for step in range(g-1):
        messages = [(rank, (rank-step) % g, x[rank][(rank-step) % g]) for rank in range(g)]
        for rank, block, value in messages:
            x[(rank+1) % g][block] += value
    for step in range(g-1):
        messages = [(rank, (rank+1-step) % g, x[rank][(rank+1-step) % g]) for rank in range(g)]
        for rank, block, value in messages:
            x[(rank+1) % g][block] = value
    return x


def two_stage_pipeline(chunks, transfer, compute):
    """仅 dispatch 与计算两个独立资源；未包含返回 combine。"""
    transfer_end = compute_end = 0.
    for _ in range(chunks):
        transfer_end += transfer
        compute_end = max(transfer_end, compute_end) + compute
    return compute_end


def checks():
    reference = (255.375, 60., 17.84375) if MODEL == 'pro' else (178.375, 42., 11.640625)
    assert tuple(x / 2**20 for x in cache_bytes(32768)) == reference
    expected_gib = 765.59765625 if MODEL == 'pro' else 137.0625
    assert routed_storage() / 2**30 == expected_gib
    # 每个窗口边界之后，按 slot=t%128 还原的最后128个 token 必须连续。
    ring = [None] * 128
    for t in range(263):
        ring[t % 128] = t
        ordered = [ring[i % 128] for i in range(max(0, t-127), t+1)]
        assert ordered == list(range(max(0, t-127), t+1))
    g = CONFIG['tp']
    rows = [[rank*10 + block - 7 for block in range(g)] for rank in range(g)]
    expected = [sum(row[block] for row in rows) for block in range(g)]
    assert ring_all_reduce(rows) == [expected] * g
    assert two_stage_pipeline(4, 2., 3.) == 14.
    assert two_stage_pipeline(4, 3., 2.) == 14.
    assert math.isclose(16/11, 1.4545454545454546)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=int, default=32768)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--capacity', type=float, default=80.)
    args = parser.parse_args()
    if min(args.context, args.batch, args.capacity) <= 0:
        parser.error('context、batch、capacity 必须为正数')
    checks()
    total = sum(cache_bytes(args.context))
    weight = routed_storage()
    eligible = [g for g in valid_tp() if weight/g <= args.capacity*2**30]
    minimum = min(eligible) if eligible else '无合法 TP 满足该下界'
    print(f'cache @ {args.context}: {total/2**20:.6f} MiB/request/rank')
    print(f'routed FP4+scale: {weight/2**30:.6f} GiB total')
    print(f'minimum TP for routed weights on {args.capacity:g} GiB: {minimum}')
    print('all checks passed')
    print(f'cache allocated for {args.batch} requests: {total*args.batch/2**30:.6f} GiB/rank')
    g, d = CONFIG['tp'], CONFIG['d']
    payload = 2*(g-1)/g*32*d*4
    print(f'one O all-reduce, B=32: {payload/1024:.3f} KiB sent/rank (ring model)')
    print('4 dispatch+compute waves: 14 ms vs 20 ms sequential (teaching simulation)')
    print('容量下界尚未计其他权重、MTP、频率表、workspace 与 allocator。')


if __name__ == '__main__':
    main()
