#!/usr/bin/env python3
"""CPU 教学实验：单提交线程 + 单 GPU stream 的离散事件模型。

python3 lab.py --host 3 --gpu 8 --kernels 4
python3 lab.py --host 8 --gpu 2
没有实际 GPU 工作：所有时间单位均为模型中的微秒，不是机器跑分。
"""
import argparse
import math


def simulate(host_us, gpu_us, kernels):
    host = device = 0.0
    rows = []
    for index in range(kernels):
        host += host_us
        start = max(host, device)
        rows.append(dict(kernel=index + 1, enqueue=host, start=start,
                         finish=start + gpu_us, queue=start - host,
                         idle=max(0.0, host - device)))
        device = start + gpu_us
    return rows


def nearest_rank(samples, percentile):
    if not samples or not 0 < percentile <= 1:
        raise ValueError('need samples and percentile in (0,1]')
    return sorted(samples)[math.ceil(percentile * len(samples)) - 1]


def amdahl(fraction, factor):
    return 1 / ((1 - fraction) + fraction / factor)


def check():
    rows = simulate(3, 8, 4)
    assert [r['finish'] for r in rows] == [11, 19, 27, 35]
    assert [r['queue'] for r in rows] == [0, 5, 10, 15]
    assert simulate(8, 2, 4)[-1]['finish'] == 34
    # 独立闭式解：常量 h、g，完成时间 = h+g+(N-1)*max(h,g)。
    for host in (1, 3, 8):
        for gpu in (1, 5, 9):
            for n in (1, 4, 20):
                assert simulate(host, gpu, n)[-1]['finish'] == host + gpu + (n-1)*max(host, gpu)
    assert nearest_rank([20]*99 + [200], .99) == 20
    assert nearest_rank([20]*95 + [45]*5, .99) == 45
    assert math.isclose(amdahl(.3, 3), 1.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', type=float, default=3)
    parser.add_argument('--gpu', type=float, default=8)
    parser.add_argument('--kernels', type=int, default=4)
    args = parser.parse_args()
    if args.host <= 0 or args.gpu <= 0 or args.kernels < 1:
        parser.error('host, gpu and kernels must be positive')
    check()
    rows = simulate(args.host, args.gpu, args.kernels)
    end, host_return = rows[-1]['finish'], rows[-1]['enqueue']
    print(f'finish={end:g} us; host_return={host_return:g} us; remaining={end-host_return:g} us')
    print('kernel enqueue start finish queue idle [us]')
    for row in rows:
        print(' '.join(f'{value:g}' for value in row.values()))
    print(f'GPU busy fraction from t=0: {args.kernels*args.gpu/end:.2%}')
    samples = [10] * 95 + [100] * 5
    print(f'tail example: mean={sum(samples)/len(samples):.1f} ms; '
          f'p50={nearest_rank(samples,.5)}; p99={nearest_rank(samples,.99)}')
    print('checks: PASS')


if __name__ == '__main__':
    main()
