#!/usr/bin/env python3
"""RMSNorm 的 CPU 正确性实验，依赖 NumPy。

python3 lab.py --width 7 --eps 1e-5
检查 dx 和 dgamma 的中心差分、零输入与非 2 次幂宽度。
实验只验证数学与精度，不把 CPU 时间当成 GPU kernel 基准。
"""
import argparse
import numpy as np


def forward(x, gamma, eps):
    r = 1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * r * gamma, r


def backward(x, gamma, dy, r):
    u = dy * gamma
    dot = np.mean(u * x, axis=-1, keepdims=True)
    dx = r * u - (r ** 3) * x * dot
    dgamma = np.sum(dy * x * r, axis=0)
    return dx, dgamma


def central_difference(x, gamma, dy, eps):
    result = []
    for array in (x, gamma):
        grad = np.zeros_like(array)
        for index in np.ndindex(array.shape):
            old, h = array[index], 1e-6
            array[index] = old + h
            plus = np.sum(forward(x, gamma, eps)[0] * dy)
            array[index] = old - h
            minus = np.sum(forward(x, gamma, eps)[0] * dy)
            array[index] = old
            grad[index] = (plus - minus) / (2 * h)
        result.append(grad)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=7)
    parser.add_argument('--eps', type=float, default=1e-5)
    args = parser.parse_args()
    if not 1 <= args.width <= 256 or not 0 < args.eps <= 1:
        parser.error('width 应在 1..256；eps 应在 (0, 1]')
    rng = np.random.default_rng(11)
    x = rng.normal(size=(3, args.width))
    gamma = rng.normal(size=args.width)
    dy = rng.normal(size=x.shape)
    y, r = forward(x, gamma, args.eps)
    dx, dg = backward(x, gamma, dy, r)
    ndx, ndg = central_difference(x, gamma, dy, args.eps)
    np.testing.assert_allclose(dx, ndx, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(dg, ndg, atol=1e-7, rtol=1e-6)
    print(f'dx max error={np.max(np.abs(dx-ndx)):.2e}; dgamma max error={np.max(np.abs(dg-ndg)):.2e}')
    # x=[1,2], gamma=[1,2] 的手算答案，与随机测试相互独立。
    hand_y, hand_r = forward(np.array([[1., 2.]]), np.array([1., 2.]), 0)
    np.testing.assert_allclose(hand_y, [[2 / np.sqrt(10), 8 / np.sqrt(10)]])
    hand_dx, hand_dg = backward(np.array([[1., 2.]]), np.array([1., 2.]), np.array([[3., -1.]]), hand_r)
    np.testing.assert_allclose(hand_dx, [[3.2 / np.sqrt(2.5), -1.6 / np.sqrt(2.5)]])
    np.testing.assert_allclose(hand_dg, [3 / np.sqrt(2.5), -2 / np.sqrt(2.5)])
    # 模拟向 2 次幂填充，归约除数必须仍是实际 width。
    block = 1 << (args.width - 1).bit_length()
    padded = np.pad(x, ((0, 0), (0, block - args.width)))
    padded_r = 1 / np.sqrt(np.sum(padded * padded, axis=1, keepdims=True) / args.width + args.eps)
    np.testing.assert_allclose(x * padded_r * gamma, y)
    zero_y, zero_r = forward(np.zeros_like(x), gamma, args.eps)
    zero_dx, zero_dg = backward(np.zeros_like(x), gamma, dy, zero_r)
    assert np.isfinite(zero_dx).all()
    np.testing.assert_array_equal(zero_y, np.zeros_like(x))
    np.testing.assert_array_equal(zero_dg, np.zeros_like(gamma))
    # 平方前升级到 FP32：FP16(300)**2 本来会超出 65504。
    big = np.array([[300., -300.]], dtype=np.float16).astype(np.float32)
    big_y, _ = forward(big, np.ones(2, dtype=np.float32), args.eps)
    assert np.isfinite(big_y).all()
    print(f'PASS: hand calculation, finite differences, width={args.width}/block={block}, zero, FP32 square')


if __name__ == '__main__':
    main()
