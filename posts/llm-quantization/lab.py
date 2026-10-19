#!/usr/bin/env python3
"""INT4 的 CPU 教学实验：对称量化、打包、误差与二阶补偿。

python3 lab.py --group-size 128
仅使用 Python 标准库。模拟整数格式与线性层误差，不测 GPU kernel。
"""
import argparse
import math


def quantize(values, scale):
    # 对称窄范围 INT4: [-7,7]；round 使用 ties-to-even。
    return [max(-7, min(7, round(v / scale))) for v in values]


def pack(q):
    if len(q) % 2:
        raise ValueError('这个小实验一次打包偶数个元素')
    return bytes((q[i] & 15) | ((q[i + 1] & 15) << 4) for i in range(0, len(q), 2))


def unpack(blob):
    out = []
    for b in blob:
        for u in [b & 15, b >> 4]:
            out.append(u - 16 if u >= 8 else u)
    return out


def quadratic_error(delta, h):
    return sum(delta[i] * h[i][j] * delta[j] for i in range(2) for j in range(2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group-size', type=int, default=128)
    args = parser.parse_args()
    if args.group_size <= 0:
        parser.error('group-size 必须为正整数')

    values, scale = [-0.875, -0.3125, 0.1875, 0.875], 0.125
    q = quantize(values, scale)
    reconstructed = [x * scale for x in q]
    assert unpack(pack(q)) == q
    assert pack([-7, 7]) == bytes([0x79])  # 独立指定字节布局。
    assert unpack(bytes([0xF8])) == [-8, -1]  # 解码支持完整 4-bit 补码。
    assert max(abs(x - y) for x, y in zip(values, reconstructed)) <= scale / 2 + 1e-12
    print('INT4 codes:', q, '; packed hex:', pack(q).hex())
    print('dequantized:', [round(v, 3) for v in reconstructed])

    # g 个权重共用 2-byte scale；先假设行宽可整除 g。
    bpw = 4 + 16 / args.group_size
    print(f'g={args.group_size}; effective bits/weight={bpw:.4f}; FP16 compression={16 / bpw:.4f}x')
    assert math.isclose(4 + 16 / 128, 4.125)

    # min delta^T H delta，第一维固定 delta0=-0.4。
    h = [[1.0, 0.5], [0.5, 1.0]]
    naive = [-0.4, 0.0]
    compensated = [-0.4, 0.2]
    loss0, loss1 = quadratic_error(naive, h), quadratic_error(compensated, h)
    # 独立网格枚举检查条件最优点，而不是再次调用补偿公式。
    candidates = [(quadratic_error([-0.4, j / 1000], h), j / 1000) for j in range(-1000, 1001)]
    assert math.isclose(min(candidates)[1], 0.2)
    assert math.isclose(loss0, 0.16) and math.isclose(loss1, 0.12)
    print(f'one-column compensation objective: {loss0:.4f} -> {loss1:.4f}')

    # W' = W diag(s), x' = diag(s)^-1 x；浮点下必须等价。
    w, x, s = [0.3, 0.9], [10.0, 1.0], [2.0, 1.0]
    original = sum(a * b for a, b in zip(w, x))
    transformed = sum(a * c * (b / c) for a, b, c in zip(w, x, s))
    assert math.isclose(original, transformed)
    print(f'channel scaling identity: {original:.4f} == {transformed:.4f}')
    print('packing, rounding bound, metadata, compensation, scaling: PASS')


if __name__ == '__main__':
    main()
