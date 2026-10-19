#!/usr/bin/env python3
"""CPU 教学模拟：分页、共享尾块、COW、引用计数；python3 lab.py --block-size 4。"""
import argparse
from collections import Counter
from math import ceil


class Pool:
    def __init__(self, capacity, size):
        self.size = size
        self.free = list(range(capacity - 1, -1, -1))
        self.data = {}
        self.refs = Counter()

    def alloc(self, initial=()):
        if not self.free:
            raise MemoryError("没有空闲块；调度器应等待或抢占")
        block = self.free.pop()
        self.data[block] = list(initial)
        self.refs[block] = 1
        return block

    def append(self, table, token):
        if not table or len(self.data[table[-1]]) == self.size:
            table.append(self.alloc())
        elif self.refs[table[-1]] > 1:
            old = table[-1]
            # 先成功分配，随后再改引用和映射；OOM 不破坏旧状态。
            new = self.alloc(self.data[old])
            self.refs[old] -= 1
            table[-1] = new
        self.data[table[-1]].append(token)

    def fork(self, table):
        for block in table:
            self.refs[block] += 1
        return table.copy()

    def release(self, table):
        for block in table:
            self.refs[block] -= 1
            if self.refs[block] == 0:
                del self.refs[block]
                del self.data[block]
                self.free.append(block)
        table.clear()

    def read(self, table):
        return [x for b in table for x in self.data[b]]

    def check(self, tables):
        assert self.refs == Counter(b for t in tables for b in t)
        assert set(self.free).isdisjoint(self.data)
        assert len(self.free) == len(set(self.free))
        assert all(0 < len(v) <= self.size for v in self.data.values())


def self_test(size):
    pool = Pool(12, size)
    a = []
    prefix = list(range(size + max(1, size - 1)))
    for token in prefix:
        pool.append(a, token)
    b = pool.fork(a)
    pool.append(a, 1001)
    pool.append(b, 2001)
    assert pool.read(a) == prefix + [1001]
    assert pool.read(b) == prefix + [2001]
    pool.check([a, b])
    pool.release(a)
    assert pool.read(b) == prefix + [2001]
    pool.check([b])
    pool.release(b)
    assert len(pool.free) == 12 and not pool.data
    # 唯一物理块仍有两个读者；COW 分配失败应完全保持原状态。
    tiny = Pool(1, 4)
    x = []
    tiny.append(x, 7)
    y = tiny.fork(x)
    before = (x.copy(), y.copy(), tiny.refs.copy(), tiny.read(x))
    try:
        tiny.append(x, 8)
    except MemoryError:
        pass
    else:
        raise AssertionError("应触发 OOM")
    assert before == (x, y, tiny.refs, tiny.read(x))
    tiny.check([x, y])
    # 与直接展平布局比较地址；K/V 分开、物理块连续的教学布局。
    table, block_size, heads, dim, t, h, d = [5, 1, 7], 4, 2, 4, 6, 1, 3
    p, offset = table[t // block_size], t % block_size
    element = ((p * block_size + offset) * heads + h) * dim + d
    assert element == 55 and 2 * element == 110


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--lengths", default="17,32,63")
    args = parser.parse_args()
    if args.block_size <= 0:
        parser.error("block-size 必须为正")
    lengths = [int(x) for x in args.lengths.split(",")]
    if any(x < 0 for x in lengths):
        parser.error("长度必须非负")
    self_test(args.block_size)
    blocks = sum(ceil(n / args.block_size) for n in lengths)
    slots = blocks * args.block_size
    print(f"block_size={args.block_size} blocks={blocks} slots={slots}")
    print(f"useful={sum(lengths)} tail_waste={slots - sum(lengths)}")
    print("PASS: dense oracle, COW isolation, refcounts, release, transactional OOM, address=110 B")


if __name__ == "__main__":
    main()
