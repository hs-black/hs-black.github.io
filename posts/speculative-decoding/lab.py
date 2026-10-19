#!/usr/bin/env python3
"""投机采样的 CPU 教学实验：概率守恒、输出长度与 KV 提交。

运行：python3 lab.py --alpha 0.8 --length 4
只使用标准库；这里的时间参数是教学假设，不是 GPU benchmark。
"""
import argparse
import math
import random


def draw(prob, rng):
    u = rng.random()
    total = 0.0
    for i, value in enumerate(prob):
        total += value
        if u < total:
            return i
    return len(prob) - 1


def residual(p, q):
    mass = [max(a - b, 0.0) for a, b in zip(p, q)]
    z = math.fsum(mass)
    return [v / z for v in mass] if z else None


def speculative_one(p, q, rng):
    """p、q 必须是相同前缀、词表及实际采样变换后的归一化分布。"""
    x = draw(q, rng)
    if rng.random() < min(1.0, p[x] / q[x]):
        return x
    return draw(residual(p, q), rng)


def expected_tokens(alpha, length):
    # 输出 = 连续接受的草稿数 + 1 个纠正或 bonus token。
    return math.fsum(alpha ** j for j in range(length + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--length', type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1 or not 0 <= args.length <= 64:
        parser.error('alpha 必须在 [0,1]；length 必须在 [0,64]')

    p, q = [0.2, 0.5, 0.3], [0.5, 0.3, 0.2]
    accepted = [min(a, b) for a, b in zip(p, q)]
    rejected = 1 - math.fsum(accepted)
    r = residual(p, q)
    exact = [a + rejected * b for a, b in zip(accepted, r)]
    assert all(math.isclose(a, b, abs_tol=1e-14) for a, b in zip(exact, p))
    assert residual(p, p) is None
    assert residual([1, 0], [0, 1]) == [1, 0]
    print('exact output distribution:', [round(v, 6) for v in exact])

    # 单独的随机实验检查采样实现；精确概率检查不依赖随机容差。
    rng = random.Random(7)
    count = [0] * len(p)
    for _ in range(100_000):
        count[speculative_one(p, q, rng)] += 1
    observed = [v / 100_000 for v in count]
    assert max(abs(a - b) for a, b in zip(observed, p)) < 0.01
    print('sampled distribution:', [round(v, 4) for v in observed])

    # 独立枚举首个拒绝位置，检验尾和公式；全部接受时多输出 bonus。
    a, n = args.alpha, args.length
    enum = sum((j + 1) * a**j * (1 - a) for j in range(n)) + (n + 1) * a**n
    expected = expected_tokens(a, n)
    assert math.isclose(enum, expected, abs_tol=1e-12)
    round_ms = 2 + 10 + 0.5 * n
    print(f'E[output]={expected:.4f}; round={round_ms:.2f} ms; per-token={round_ms / expected:.4f} ms')
    print(f'speedup vs 10 ms/token={10 * expected / round_ms:.4f}x')

    # target KV 已含原前缀和候选；新采样的纠正 token 尚未过前向。
    target_kv = ['A', 'B', 'E', 'F', 'G', 'H']
    prefix_len, accepted_len = 2, 2
    target_kv = target_kv[:prefix_len + accepted_len]
    output = target_kv + ['X']
    assert output == ['A', 'B', 'E', 'F', 'X']
    assert len(output) == len(target_kv) + 1
    target_kv.append(output[-1])  # 下一轮输入 anchor X，才能提交 X 的 KV。
    assert target_kv == output
    print('probability, length enumeration, and KV alignment: PASS')


if __name__ == '__main__':
    main()
