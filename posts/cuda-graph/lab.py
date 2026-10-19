#!/usr/bin/env python3
"""CPU 教学模拟：固定存储、别名、shape bucket 和 capture 摊销。

python3 lab.py --replays 300
真实 CUDA 捕获见同目录 gpu_example.py；此文件不调用 CUDA。
"""
import argparse
import math


class RecordedSquare:
    """保留输入/输出列表引用，模拟 replay 读写固定存储。"""
    def __init__(self, source):
        self.source = source
        self.output = [0 for _ in source]

    def replay(self):
        for i, value in enumerate(self.source):
            self.output[i] = value * value + 1
        return self.output


def bucket_slots(batches, buckets):
    buckets = sorted(buckets)
    selected = []
    for batch in batches:
        fits = [b for b in buckets if b >= batch]
        if not fits:
            raise ValueError(f'no bucket for batch={batch}')
        selected.append(fits[0])
    return selected


def first_profitable(capture_us, eager_us, graph_us):
    delta = eager_us - graph_us
    return math.floor(capture_us / delta) + 1 if delta > 0 else None


def check():
    source = [3]
    graph = RecordedSquare(source)
    alias = graph.replay()
    saved = alias.copy()
    source[0] = 4
    graph.replay()
    assert alias == [17] and saved == [10]
    # 重绑定局部变量不会更改 graph 保留的引用。
    source = [9]
    assert graph.replay() == [17]
    assert source == [9]
    assert bucket_slots([1, 3, 5, 8], [1, 2, 4, 8]) == [1, 4, 8, 8]
    assert first_profitable(600, 12, 10) == 301
    assert first_profitable(600, 12, 12) is None
    assert first_profitable(1200, 30, 24) == 201


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replays', type=int, default=300)
    args = parser.parse_args()
    if args.replays < 1:
        parser.error('replays must be positive')
    check()
    source = [3]
    graph = RecordedSquare(source)
    alias = graph.replay()
    saved = alias.copy()
    source[:] = [4]
    graph.replay()
    slots = sum(bucket_slots([1, 3, 5, 8], [1, 2, 4, 8]))
    print(f'alias={alias[0]}; clone={saved[0]}; slots={slots}; break_even={first_profitable(600,12,10)}')
    print(f'N={args.replays}: eager={args.replays*12} us; graph={600+args.replays*10} us')
    print(f'padding waste={(slots-17)/slots:.2%}')
    print('checks: PASS')


if __name__ == '__main__':
    main()
