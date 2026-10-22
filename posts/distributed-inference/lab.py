#!/usr/bin/env python3
"""CPU 教学实验：验证 TP 等价性，计算通信和流水线预算。

运行：python3 lab.py --ranks 4 --microbatches 8
先预测再修改：把 W1 的列改掉；把 ranks 改成 8；把 microbatches 改成 1。
这里只用小矩阵和成本模型，不模拟 NCCL 的实际带宽或算法选择。
"""
import argparse
import math


def matmul(a, b):
    assert len(a[0]) == len(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def relu(a):
    return [[max(0, x) for x in row] for row in a]


def split_mlp(x, w1, w2, ranks=2):
    """先切输出通道，再切下一层输入通道，最后归约同位置部分和。"""
    width = len(w2)
    if width % ranks:
        raise ValueError('FFN width must be divisible by ranks')
    chunk = width // ranks
    partials = []
    for rank in range(ranks):
        left, right = rank * chunk, (rank + 1) * chunk
        w1_local = [row[left:right] for row in w1]
        w2_local = w2[left:right]
        partials.append(matmul(relu(matmul(x, w1_local)), w2_local))
    return [[sum(part[i][j] for part in partials)
             for j in range(len(w2[0]))] for i in range(len(x))]


def ring_model(message_bytes, ranks, alpha_us=3.0, bandwidth_gbs=100.0):
    sent = 2 * (ranks - 1) / ranks * message_bytes
    time_us = 2 * (ranks - 1) * alpha_us + sent / (bandwidth_gbs * 1e9) * 1e6
    return sent, time_us


def pipeline(stages, microbatches, stage_ms=2.0):
    ticks = microbatches + stages - 1
    return ticks * stage_ms, microbatches / ticks


def check():
    # 手算答案独立于矩阵函数：11*[1,2] + 17*[3,4] = [62,90]。
    x, w1, w2 = [[1, 2]], [[3, 5], [4, 6]], [[1, 2], [3, 4]]
    assert split_mlp(x, w1, w2) == [[62, 90]]
    # 第二组含负数，防止“把 ReLU 移到求和后”也蒙混过关。
    x2 = [[-1, 2], [3, -4]]
    assert split_mlp(x2, w1, w2) == matmul(relu(matmul(x2, w1)), w2)
    # p=1 时无需通信；四卡 8 MiB 单卡发送 12 MiB。
    assert ring_model(1234, 1) == (0, 0)
    sent, time_us = ring_model(8 * 2**20, 4)
    assert sent == 12 * 2**20 and math.isclose(time_us, 143.82912)
    total, util = pipeline(4, 12)
    assert total == 30 and math.isclose(util, 0.8)
    assert pipeline(4, 1) == (8, 0.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', type=int, default=4)
    parser.add_argument('--microbatches', type=int, default=8)
    args = parser.parse_args()
    if args.ranks < 1 or args.microbatches < 1:
        parser.error('ranks and microbatches must be positive')
    check()
    result = split_mlp([[1, 2]], [[3, 5], [4, 6]], [[1, 2], [3, 4]])[0]
    sent, time_us = ring_model(8 * 2**20, args.ranks)
    total, util = pipeline(4, args.microbatches)
    print(f'TP result = {result}; ring = {sent / 2**20:.2f} MiB; PP = {util:.2%}')
    print(f'ring estimate = {time_us:.3f} us; pipeline total = {total:.1f} ms')
    # 64B 总参数，而不是 8B 每 token 激活参数，决定常驻权重。
    weight_gib = 64e9 * 2 / 4 / 2**30
    print(f'capacity: weights={weight_gib:.2f} GiB; with KV/workspace={weight_gib+11:.2f} GiB')
    print('checks: PASS')


if __name__ == '__main__':
    main()
