#!/usr/bin/env python3
"""V4-Pro 工程小实验：分组投影、mHC、路由权重和专家参数量。
python3 lab.py；NumPy 小张量验证，不加载真实模型。
练习：改 groups 或 rank，先预测参数量；改 bias 验证只改变选择过程。
"""
import numpy as np


def sinkhorn(logits, steps=100):
    z = np.exp(logits - logits.max())
    for _ in range(steps):
        z /= z.sum(axis=1, keepdims=True)
        z /= z.sum(axis=0, keepdims=True)
    return z


def main():
    rng = np.random.default_rng(42)
    # 真模型是 16 groups，每组 4096 -> 1024；教学缩成 4 groups，6 -> 2。
    groups, width, rank, output = 4, 6, 2, 7
    x = rng.normal(size=(3, groups, width))
    wa = rng.normal(size=(groups, rank, width))
    wb = rng.normal(size=(groups * rank, output))
    grouped = np.einsum("bgd,grd->bgr", x, wa).reshape(3, -1) @ wb
    # 独立构造 block-diagonal dense 权重，与分组 einsum 对照。
    dense = np.zeros((groups * width, groups * rank))
    for group in range(groups):
        dense[group*width:(group+1)*width, group*rank:(group+1)*rank] = wa[group].T
    reference = x.reshape(3, -1) @ dense @ wb
    np.testing.assert_allclose(grouped, reference, atol=1e-12)
    b = sinkhorn(rng.normal(size=(4, 4)))
    np.testing.assert_allclose(b.sum(0), np.ones(4), atol=1e-10)
    np.testing.assert_allclose(b.sum(1), np.ones(4), atol=1e-10)
    # 明确的 doubly-stochastic 矩阵，复核幻灯片上的手算。
    residual = np.array([1., 3., 5., 7.])
    pre, post = np.array([.1, .2, .3, .4]), np.array([.25, .5, .75, 1.])
    combination = .5*np.eye(4) + .125*np.ones((4, 4))
    y = combination.T @ residual + post*(2*(pre @ residual))
    np.testing.assert_allclose(y, [5., 8.5, 12., 15.5])
    scores, bias = np.array([1., 2., 3., 4.]), np.array([4., 0., 0., 0.])
    ids = np.argsort(-(scores+bias), kind="stable")[:2]
    weights = scores[ids]/scores[ids].sum()*2.5
    assert ids.tolist() == [0, 3]
    np.testing.assert_allclose(weights, [.5, 2.])
    assert 3*7168*3072 == 66_060_288
    assert 61*384*3*7168*3072 == 1_547_396_186_112
    print("grouped projection: equal")
    print("Sinkhorn max error < 1e-10")
    print("routing weights:", weights.tolist())
    print("all checks passed")


if __name__ == "__main__":
    main()
