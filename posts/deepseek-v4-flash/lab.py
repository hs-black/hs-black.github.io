#!/usr/bin/env python3
"""V4-Flash 的小张量实验：重叠压缩、因果边界、路由和 mHC。
运行 python3 lab.py；只依赖 NumPy，无需模型权重或 GPU。
可改 values、gate 或 route bias；先预测，随后对照 batch/streaming 两条路径。
"""
import numpy as np


def pool(values, logits):
    """每个通道独立在 token 轴做 softmax；这一步在 RMSNorm/RoPE 之前。"""
    z = logits - logits.max(axis=0, keepdims=True)
    w = np.exp(z)
    w /= w.sum(axis=0, keepdims=True)
    return (values * w).sum(axis=0)


def prefill_overlap(v, g, ratio=4):
    """v/g: [tokens, 2*channels]，前半给下一组重叠、后半给当前组。"""
    d = v.shape[1] // 2
    out = []
    for end in range(ratio, len(v) + 1, ratio):
        now_v, now_g = v[end-ratio:end, d:], g[end-ratio:end, d:]
        if end > ratio:
            now_v = np.concatenate([v[end-2*ratio:end-ratio, :d], now_v])
            now_g = np.concatenate([g[end-2*ratio:end-ratio, :d], now_g])
        out.append(pool(now_v, now_g))
    return np.array(out)


def streaming_overlap(v, g, ratio=4):
    d = v.shape[1] // 2
    previous_v = previous_g = None
    current_v, current_g, out = [], [], []
    for t, (value, gate) in enumerate(zip(v, g)):
        current_v.append(value)
        current_g.append(gate)
        if (t + 1) % ratio:
            continue
        cv, cg = np.array(current_v), np.array(current_g)
        pv, pg = cv[:, d:], cg[:, d:]
        if previous_v is not None:
            pv = np.concatenate([previous_v[:, :d], pv])
            pg = np.concatenate([previous_g[:, :d], pg])
        out.append(pool(pv, pg))
        previous_v, previous_g = cv, cg
        current_v, current_g = [], []
    return np.array(out)


def route(scores, bias, scale):
    ids = np.argsort(-(scores + bias), kind="stable")[:2]
    weights = scores[ids] / scores[ids].sum() * scale
    return ids, weights


def main():
    rng = np.random.default_rng(42)
    v, g = rng.normal(size=(15, 6)), rng.normal(size=(15, 6))
    batched, streamed = prefill_overlap(v, g), streaming_overlap(v, g)
    np.testing.assert_allclose(batched, streamed, atol=1e-12)
    # 独立可手算例：均匀 gate 时，第2组把前组前半和当前组后半取均值。
    simple = np.arange(16, dtype=float).reshape(8, 2)
    answer = prefill_overlap(simple, np.zeros_like(simple))
    np.testing.assert_allclose(answer[:, 0], [4, 7.5])
    scalar = pool(np.array([2., 4., 8., 10.]), np.log([.1, .2, .3, .4]))
    np.testing.assert_allclose(scalar, 7.4)
    for t, expected in [(2, []), (3, [0]), (6, [0]), (7, [0, 1])]:
        visible = np.arange(4)[np.arange(4) < (t+1)//4].tolist()
        assert visible == expected
    ids, weights = route(np.array([1., 2., 3., 4.]), np.array([4., 0., 0., 0.]), 1.5)
    assert ids.tolist() == [0, 3]
    np.testing.assert_allclose(weights, [.3, 1.2])
    x, pre, post = np.array([2., 4., 6., 8.]), np.array([.1, .2, .3, .4]), np.array([.25, .5, .75, 1.])
    z = pre @ x
    y = np.eye(4).T @ x + post * (2*z)
    np.testing.assert_allclose(y, [5., 10., 15., 20.])
    print("prefill == streaming: True")
    print(f"pooled scalar: {scalar:.6f}")
    print("routing weights:", np.round(weights, 3).tolist())
    print("all checks passed")


if __name__ == "__main__":
    main()
