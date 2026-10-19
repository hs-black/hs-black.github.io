#!/usr/bin/env python3
"""激活重算的 CPU 教学实现（NumPy）。

python3 lab.py --layers 12 --segment 3
模型：h[i+1] = tanh(h[i] @ W[i])；loss = sum(h[-1] * target)。
对照完整保存版与手写分段重算版，并用中心差分独立验证梯度。
peak_slots 计不同数组的激活张量数，不包含参数、梯度、临时 matmul
工作区、Python 对象或框架分配器，不能拿它当真实 GPU 显存。
"""
import argparse
import numpy as np


def full(x, weights, target):
    hs = [x]
    for w in weights:
        hs.append(np.tanh(hs[-1] @ w))
    grad = target.copy()
    dw = [None] * len(weights)
    for i in reversed(range(len(weights))):
        dz = grad * (1 - hs[i + 1] ** 2)
        dw[i] = hs[i].T @ dz
        grad = dz @ weights[i].T
    return hs[-1], grad, dw


def segmented(x, weights, target, segment):
    saved, h, n = {}, x, len(weights)
    for i, w in enumerate(weights):
        if i % segment == 0:
            saved[i] = h
        h = np.tanh(h @ w)
    output = h.copy()
    del h
    grad, dw, replayed, peak = target.copy(), [None] * n, 0, len(saved)
    for start in reversed(range(0, n, segment)):
        end = min(start + segment, n)
        hs = [saved.pop(start)]
        for i in range(start, end):
            hs.append(np.tanh(hs[-1] @ weights[i]))
            replayed += 1
            # 被弹出的边界在 hs[0]，不重复计数；输出另计 1。
            peak = max(peak, len(saved) + len(hs) + 1)
        for i in reversed(range(start, end)):
            dz = grad * (1 - hs[-1] ** 2)
            dw[i] = hs[-2].T @ dz
            grad = dz @ weights[i].T
            hs.pop()
        del hs
    return output, grad, dw, replayed, peak


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layers', type=int, default=12)
    p.add_argument('--segment', type=int, default=3)
    args = p.parse_args()
    if not 1 <= args.layers <= 64 or not 1 <= args.segment <= args.layers:
        p.error('要求 1 <= segment <= layers <= 64')
    rng = np.random.default_rng(19)
    x = rng.normal(size=(2, 5))
    weights = [np.eye(5) * .8 + rng.normal(scale=.1, size=(5, 5)) for _ in range(args.layers)]
    target = rng.normal(size=(2, 5))
    y, dx, dw = full(x, weights, target)
    cy, cdx, cdw, replayed, peak = segmented(x, weights, target, args.segment)
    np.testing.assert_allclose(cy, y, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(cdx, dx, atol=1e-12, rtol=1e-12)
    for a, b in zip(cdw, dw):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
    # 对每层抽一个参数做差分；避免只比较两份同样的反向公式。
    errors = []
    for i, w in enumerate(weights):
        old, h = w[1, 3], 1e-6
        w[1, 3] = old + h
        plus = np.sum(full(x, weights, target)[0] * target)
        w[1, 3] = old - h
        minus = np.sum(full(x, weights, target)[0] * target)
        w[1, 3] = old
        errors.append(abs((plus - minus) / (2 * h) - cdw[i][1, 3]))
    assert max(errors) < 1e-7
    assert replayed == args.layers
    bound = (args.layers + args.segment - 1) // args.segment + args.segment + 1
    assert peak <= bound
    print(f'layers={args.layers}, segment={args.segment}, replayed={replayed}')
    print(f'activation peak: full={args.layers + 1} slots; checkpoint={peak} slots (output included)')
    print(f'gradient finite-difference max error={max(errors):.2e}')
    # 一次 dropout 的重算必须复用掩码。随机状态恢复后掩码完全相同。
    saved_rng = rng.bit_generator.state
    mask = rng.random((16, 16)) > .5
    wrong_mask = rng.random((16, 16)) > .5
    rng.bit_generator.state = saved_rng
    replay_mask = rng.random((16, 16)) > .5
    assert np.array_equal(mask, replay_mask) and not np.array_equal(mask, wrong_mask)
    print('PASS: output/grad parity, independent finite differences, replay count, RNG replay')


if __name__ == '__main__':
    main()
