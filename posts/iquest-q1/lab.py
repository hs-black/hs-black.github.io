#!/usr/bin/env python3
"""IQuest-Q1 numerical mechanisms on a CPU (NumPy only).

Run from this directory:
    python -m pip install numpy
    python lab.py
    python lab.py --tokens 524288 --batch 4 --kv-bytes 1

The float64 experiments verify sink attention, LSE merging and the ordinary
Stable SplitNorm block's residual wiring. Small tensors and weights are
teaching examples. The KV ledger uses the published backbone dimensions and
counts logical K/V payload; weights, page metadata, workspace and MTP are
separate allocations. No checkpoint, accelerator, model API or dataset is
required. Timing and GPU throughput are not measured.
"""

from __future__ import annotations

import argparse
import numpy as np


def softmax_lse(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stable softmax and log(sum(exp(scores))) along the last dimension.

    A masked score is -inf. An entirely masked row returns zero weights and
    LSE=-inf. Finite scores are required before applying the mask.
    """
    scores = np.asarray(scores, dtype=np.float64)
    maximum = np.max(scores, axis=-1, keepdims=True)
    shift = np.where(np.isfinite(maximum), maximum, 0.0)
    numerator = np.exp(scores - shift)
    denominator = numerator.sum(axis=-1, keepdims=True)
    weights = np.divide(numerator, denominator,
                        out=np.zeros_like(numerator), where=denominator > 0)
    log_denominator = np.full_like(denominator, -np.inf)
    np.log(denominator, out=log_denominator, where=denominator > 0)
    return weights, (shift + log_denominator)[..., 0]


def attention(scores: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights, lse = softmax_lse(scores)
    return weights @ values, lse


def merge_partials(outputs: list[np.ndarray], lses: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Merge disjoint key partitions; every key (including a sink) occurs once."""
    stacked_lse = np.stack(lses, axis=-1)
    partition_weights, total_lse = softmax_lse(stacked_lse)
    stacked_outputs = np.stack(outputs, axis=-2)
    result = (stacked_outputs * partition_weights[..., :, None]).sum(axis=-2)
    return result, total_lse


def add_sink(output: np.ndarray, real_lse: np.ndarray,
             sink_score: np.ndarray) -> np.ndarray:
    """Zero-V sink: output *= exp(LSE_real - logaddexp(LSE_real, score_sink)).

    This is sigmoid(LSE_real - score_sink), evaluated without exp overflow.
    sink_score is finite, so an all-masked real row receives a zero gate.
    """
    combined_lse = np.logaddexp(real_lse, sink_score)
    gate = np.exp(real_lse - combined_lse)
    return output * gate[..., None]


def sink_experiment(rng: np.random.Generator) -> None:
    # One attention head: Q=[queries, d], K=[tokens, d], V=[tokens, dv].
    q, k = rng.normal(size=(5, 8)), rng.normal(size=(7, 8))
    v, sink_key = rng.normal(size=(7, 4)), rng.normal(size=8)
    scores = q @ k.T / np.sqrt(8)
    sink_scores = q @ sink_key / np.sqrt(8)
    # Mask differently for each query, including one entirely masked row.
    mask = np.arange(7)[None, :] <= np.arange(5)[:, None]
    mask[0] = False
    scores = np.where(mask, scores, -np.inf)
    ordinary, real_lse = attention(scores, v)
    fused = add_sink(ordinary, real_lse, sink_scores)
    dense, _ = attention(np.column_stack((scores, sink_scores)),
                         np.vstack((v, np.zeros((1, 4)))))
    np.testing.assert_allclose(fused, dense, atol=2e-14, rtol=2e-14)
    np.testing.assert_array_equal(fused[0], np.zeros(4))

    # Scores of magnitude 1000 make naive exp(score) overflow.
    extreme = np.array([[1000., 999., -np.inf], [-1000., -1001., -np.inf]])
    extreme_values = np.array([[1., 2.], [3., 4.], [5., 6.]])
    extreme_sink = np.array([1001., -999.])
    out, lse = attention(extreme, extreme_values)
    stable = add_sink(out, lse, extreme_sink)
    direct, _ = attention(np.column_stack((extreme, extreme_sink)),
                          np.vstack((extreme_values, np.zeros((1, 2)))))
    assert np.isfinite(stable).all()
    np.testing.assert_allclose(stable, direct, atol=2e-13, rtol=2e-13)

    demo_scores = np.log(np.array([[2., 1., 1.]]))
    demo_values = np.array([[1.], [3.], [-1.]])
    plain, lse = attention(demo_scores, demo_values)
    gated = add_sink(plain, lse, np.log(np.array([4.])))
    np.testing.assert_allclose([plain.item(), gated.item()], [1., .5])
    print("1. Query-aware sink / zero V")
    print(f"   拼接 sink K/V 与 LSE 缩放最大误差：{np.max(np.abs(fused-dense)):.3e}")
    print("   真实键全部被 mask：输出 0；±1000 logits：结果有限且与参考一致。")
    print(f"   演示权重 [2,1,1]：原输出 {plain.item():.3f}；sink 权重 4 后 {gated.item():.3f}。")


def halo_experiment(rng: np.random.Generator) -> None:
    scores = rng.normal(size=(6, 11)) * 2
    values = rng.normal(size=(11, 5))
    scores[0, :4] = -np.inf  # A rank can have no valid keys for a boundary query.
    scores[-1, :] = -np.inf
    parts = [slice(0, 4), slice(4, 8), slice(8, 11)]
    partials = [attention(scores[:, part], values[part]) for part in parts]
    merged, _ = merge_partials([o for o, _ in partials], [lse for _, lse in partials])
    reference, lse = attention(scores, values)
    np.testing.assert_allclose(merged, reference, atol=2e-14, rtol=2e-14)
    np.testing.assert_array_equal(merged[-1], np.zeros(5))

    local, lse_local = attention(np.log(np.array([[1., 3.]])), np.array([[0.], [4.]]))
    halo, lse_halo = attention(np.log(np.array([[2.]])), np.array([[1.]]))
    correct, all_lse = merge_partials([local, halo], [lse_local, lse_halo])
    wrong = local + halo
    np.testing.assert_allclose(correct, [[7/3]])
    np.testing.assert_allclose(wrong, [[4.]])

    # A global sink of exp(score)=2 must be counted once across both partitions.
    sink = np.log(np.array([2.]))
    once = add_sink(correct, all_lse, sink)
    local_sink = add_sink(local, lse_local, sink)
    halo_sink = add_sink(halo, lse_halo, sink)
    duplicated, _ = merge_partials([local_sink, halo_sink],
                                   [np.logaddexp(lse_local, sink), np.logaddexp(lse_halo, sink)])
    np.testing.assert_allclose(once, [[1.75]])
    np.testing.assert_allclose(duplicated, [[1.4]])
    assert not np.allclose(once, duplicated)
    print("2. Halo 分块与 LSE 重归一")
    print(f"   三个分块合并与完整 attention 最大误差：{np.max(np.abs(merged-reference)):.3e}")
    print(f"   local 输出 3，halo 输出 1：直接相加 {wrong.item():.6f}，正确合并 {correct.item():.6f}。")
    print(f"   sink 只计一次：{once.item():.3f}；每个分块重复计入 sink：{duplicated.item():.3f}。")


def layer_types() -> list[str]:
    """88 backbone layers, including the full-attention boundary layers."""
    return ["FA", "FA"] + ["SWA", "SWA", "SWA", "FA"] * 21 + ["FA", "FA"]


def kv_bytes(tokens: int, batch: int = 1, bytes_per_element: int = 2) -> tuple[int, int]:
    types = layer_types()
    per_position = 2 * 8 * 128 * bytes_per_element  # K + V, KV heads, head dim, dtype.
    positions = types.count("FA") * tokens + types.count("SWA") * min(tokens, 4096)
    return batch * per_position * positions, batch * per_position * len(types) * tokens


def memory_experiment(tokens: int, batch: int, bytes_per_element: int) -> None:
    types = layer_types()
    assert len(types) == 88 and types.count("FA") == 25 and types.count("SWA") == 63
    expected_gib = {32768: 4.109375, 131072: 13.484375, 524288: 50.984375}
    print("3. 逻辑 KV payload（GiB = 2^30 bytes）")
    print("   主干：25 层 FA + 63 层 SWA；窗口 4096；8 KV heads；head dim 128。")
    for length, expected in expected_gib.items():
        mixed, all_full = kv_bytes(length)
        assert mixed / 2**30 == expected
        print(f"   B=1，BF16，T={length:>6}：混合 {mixed/2**30:9.6f} GiB；全 FA {all_full/2**30:7.3f} GiB")
    assert kv_bytes(524288)[1] / 2**30 == 176
    # Before filling the local window, all layers retain the whole prefix.
    assert kv_bytes(2048)[0] == kv_bytes(2048)[1]
    mixed, all_full = kv_bytes(tokens, batch, bytes_per_element)
    print(f"   指定配置 B={batch}, T={tokens}, {bytes_per_element} bytes/元素："
          f"混合 {mixed/2**30:.6f} GiB；全 FA {all_full/2**30:.6f} GiB。")


def rmsnorm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Unit learned gain for the teaching example."""
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def splitnorm_experiment(rng: np.random.Generator) -> None:
    x = 5 * rng.normal(size=(6, 8))
    wq, wk, wv, wo = [rng.normal(size=(8, 8)) / np.sqrt(8) for _ in range(4)]
    wg, wu = [rng.normal(size=(8, 12)) / np.sqrt(8) for _ in range(2)]
    wd = rng.normal(size=(12, 8)) / np.sqrt(12)

    def attn(z):
        scores = (z @ wq) @ (z @ wk).T / np.sqrt(8)
        scores[np.triu_indices(len(z), 1)] = -np.inf
        return attention(scores, z @ wv)[0] @ wo

    def ffn(z):
        gate, up = z @ wg, z @ wu
        return ((gate / (1 + np.exp(-gate))) * up) @ wd

    z = rmsnorm(x)
    update = rmsnorm(attn(z))
    h = z + update
    y = h + ffn(rmsnorm(h))  # Teaching FFN output scale = 1.
    np.testing.assert_allclose(h - update, z, atol=2e-14)
    np.testing.assert_allclose(y - h, ffn(rmsnorm(h)), atol=2e-14)
    wrong_h = x + update
    np.testing.assert_allclose(wrong_h - h, x - z, atol=2e-14)
    assert not np.allclose(wrong_h, h)
    pre_h = x + attn(rmsnorm(x))
    pre_y = pre_h + ffn(rmsnorm(pre_h))
    print("4. 普通 Stable SplitNorm 层的前向连线")
    print("   z=Norm(x)，h=z+Norm(Attention(z))，y=h+FFN(Norm(h))。")
    print(f"   输入 RMS={np.sqrt(np.mean(x*x)):.6f}，归一化残差 z 的 RMS={np.sqrt(np.mean(z*z)):.6f}。")
    print(f"   错接为 x+Norm(Attention(z)) 时，h 的最大差异={np.max(np.abs(wrong_h-h)):.6f}。")
    print(f"   同组教学权重：SplitNorm 输出 RMS={np.sqrt(np.mean(y*y)):.6f}；"
          f"标准 Pre-Norm 输出 RMS={np.sqrt(np.mean(pre_y*pre_y)):.6f}。")
    print("   这些数值用于检查前向公式；首层 SandwichNorm 和实际训练增益分别考察。")


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=positive_int, default=524288)
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--kv-bytes", type=int, choices=[1, 2, 4], default=2)
    args = parser.parse_args()
    rng = np.random.default_rng(20261023)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        sink_experiment(rng)
        halo_experiment(rng)
        memory_experiment(args.tokens, args.batch, args.kv_bytes)
        splitnorm_experiment(rng)
    print("PASS：sink、mask、LSE 合并、缓存账本与残差连线检查全部通过。")


if __name__ == "__main__":
    main()
