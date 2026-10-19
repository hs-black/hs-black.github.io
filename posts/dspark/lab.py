#!/usr/bin/env python3
"""DSpark 思路的 CPU 教学模拟：Markov、存活概率与固定预算调度。

python3 lab.py --budget 5
budget 是 target 输入位置总数，包含每个请求的一个 anchor。
这是解析小模型，不加载 DSpark 权重，也不测真实服务性能。
"""
import argparse
import itertools
import math


def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    z = math.fsum(exps)
    return [x / z for x in exps]


def survival(confidence):
    result, value = [], 1.0
    for c in confidence:
        value *= c
        result.append(value)
    return result


def allocate(curves, budget):
    """已知、在当前候选抽样前确定的置信度；不能拿未来 token 反选当前 token。"""
    lengths = [0] * len(curves)
    candidates = sorted(((a, r, j) for r, curve in enumerate(curves)
                         for j, a in enumerate(curve)), key=lambda item: (-item[0], item[1], item[2]))
    for _, r, j in candidates[:budget - len(curves)]:
        lengths[r] = max(lengths[r], j + 1)
    return lengths


def reward(curves, lengths):
    return len(curves) + math.fsum(math.fsum(curve[:n]) for curve, n in zip(curves, lengths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, default=5)
    args = parser.parse_args()
    if not 2 <= args.budget <= 8:
        parser.error('budget 必须在 [2,8]')

    # rank=1 的两词表 Markov head，bias = W1[prev] @ W2。
    w1 = [1, -1]
    w2 = [math.log(3) / 2, -math.log(3) / 2]
    transition = [softmax([v * w for w in w2]) for v in w1]
    assert math.isclose(transition[0][0], 0.75)
    assert math.isclose(transition[1][1], 0.75)
    mismatch = 0.5 * transition[0][1] + 0.5 * transition[1][0]
    assert math.isclose(mismatch, 0.25)
    print('Markov inconsistent-pair probability:', round(mismatch, 4))

    curves = [survival([0.9, 0.8, 0.5]), survival([0.6, 0.6, 0.6])]
    # 每个固定预算都枚举全部前缀分配，独立检查边际收益排序。
    for budget in range(2, 9):
        lengths = allocate(curves, budget)
        feasible = [ns for ns in itertools.product(range(4), repeat=2) if sum(ns) == budget - 2]
        optimum = max(reward(curves, ns) for ns in feasible)
        assert math.isclose(reward(curves, lengths), optimum)
    for boundary in [[[1, 1, 1], [1, 1, 1]], [[0, 0, 0], [0, 0, 0]], [[0.5, 0.5, 0], [0.5, 0.5, 0]]]:
        for budget in range(2, 9):
            lengths = allocate(boundary, budget)
            assert sum(lengths) == budget - 2
            feasible = [ns for ns in itertools.product(range(4), repeat=2) if sum(ns) == budget - 2]
            assert math.isclose(reward(boundary, lengths), max(reward(boundary, ns) for ns in feasible))
    lengths = allocate(curves, args.budget)
    print(f'B={args.budget}; prefixes={lengths}; E[output]={reward(curves, lengths):.4f}')

    # 合成 profile，单位 ms；真实系统应测同一硬件和请求形状。
    costs = {2: 4.0, 3: 4.3, 4: 4.7, 5: 5.4, 6: 6.4, 7: 8.0, 8: 10.0}
    rates = {b: 1000 * reward(curves, allocate(curves, b)) / t for b, t in costs.items()}
    best = max(rates, key=rates.get)
    assert best == 5
    print(f'synthetic profile optimum: B={best}; throughput={rates[best]:.2f} token/s')
    predicted = 1 + sum(survival([0.9] * 4))
    actual = 1 + sum(survival([0.8] * 4))
    print(f'overconfident E[output]: {predicted:.4f} vs {actual:.4f}')

    # p=q=(.5,.5)。只在抽中 A 时采用草稿，否则重新抽 p：输出被偏到 .75。
    biased_a = 0.5 * 1.0 + 0.5 * 0.5
    causal_a = 0.5 * 0.5 + 0.5 * 0.5  # 独立硬币决定是否采用草稿。
    assert biased_a == 0.75 and causal_a == 0.5
    print('selection-bias example: P(A)=0.75; causal decision: P(A)=0.50')
    print('Markov, exhaustive allocation, calibration, causality: PASS')


if __name__ == '__main__':
    main()
