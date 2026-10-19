#!/usr/bin/env python3
"""稀疏 gather 与 dense masked attention 的 CPU 正确性实验。

运行: python3 lab.py --seed 7 --window 4
修改 build_mask() 加入过去的固定全局位置；观察并集和因果 mask。
这不是 GPU 稀疏 kernel，也不使用 Python 循环时间预测 GPU 吞吐。
"""
import argparse
import numpy as np


def build_mask(n, window):
    q = np.arange(n)[:, None]
    k = np.arange(n)[None, :]
    return (k <= q) & (k > q - window)


def dense_masked(q, k, v, mask):
    if np.any(mask.sum(axis=-1) == 0):
        raise ValueError('每个 query 至少需要一个可见位置')
    logits = q @ k.T / np.sqrt(q.shape[-1])
    logits = np.where(mask, logits, -np.inf)
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v


def sparse_gather(q, k, v, mask):
    result = []
    for row, qi in enumerate(q):
        selected = np.flatnonzero(mask[row])
        logits = (k[selected] @ qi) / np.sqrt(k.shape[-1])
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        result.append(weights @ v[selected])
    return np.array(result)


def causal_topk(scores, position, count):
    valid = np.arange(len(scores)) <= position
    masked = np.where(valid, scores, -np.inf)
    return np.argsort(masked)[-min(count, valid.sum()):]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--window', type=int, default=4)
    args = ap.parse_args()
    if args.window < 1:
        ap.error('--window 必须为正整数')
    rng = np.random.default_rng(args.seed)
    n, d = 24, 8
    q, k = rng.normal(size=(2, n, d))
    v = rng.normal(size=(n, 5))
    mask = build_mask(n, args.window)
    ref = dense_masked(q, k, v, mask)
    np.testing.assert_allclose(ref, sparse_gather(q, k, v, mask), atol=1e-12)
    print('masked dense vs gather: PASS')
    # 未来扰动：早期 Q 固定，未来 K/V 改成很大的随机数，旧输出必须不变。
    cutoff = 13
    k2, v2 = k.copy(), v.copy()
    k2[cutoff:] *= 100
    v2[cutoff:] += 100
    changed = sparse_gather(q, k2, v2, mask)
    np.testing.assert_allclose(ref[:cutoff], changed[:cutoff], atol=1e-12)
    print('future perturbation: PASS')
    assert build_mask(8, 3).sum() == 21
    assert sum(min(i + 1, 3) for i in range(8)) == 21  # 独立逐行计数
    print('window pairs n=8,w=3: 21')
    targets = np.array([1, 6, 14])
    blocks = np.unique(targets // 4)
    loaded = len(blocks) * 4
    assert loaded == 12
    print(f'{len(targets)} targets / {loaded} loaded = {100*len(targets)/loaded:.1f}%')
    # 检查“先选未来、后删除”的反例。
    scores = np.array([1., 2., 3., 100., 101.])
    good = causal_topk(scores, position=2, count=2)
    np.testing.assert_array_equal(np.sort(good), [1, 2])
    bad = np.argsort(scores)[-2:]
    assert len(bad[bad <= 2]) == 0
    # 手算 softmax 选择 {0, 2} 的独立 oracle。
    weights = np.exp([0., np.log(3.)])
    weights /= weights.sum()
    np.testing.assert_allclose(weights @ np.array([2., 8.]), 6.5)


if __name__ == '__main__':
    main()
