#!/usr/bin/env python3
"""张量布局实验：python3 lab.py [--stride 2] [--offset 1]

依赖 NumPy。打印的是地址、32-byte sector 与字节账本，不是 GPU 延迟。
尝试改变 stride / offset，先预测 sector 数，再用逐字节参考检查验证。
"""
import argparse
import numpy as np


def address(index, strides, storage_offset=0):
    """元素偏移；PyTorch stride 单位为元素，NumPy strides 单位为 byte。"""
    return storage_offset + sum(i * s for i, s in zip(index, strides))


def sectors_formula(stride, offset=0, lanes=32, itemsize=4, sector=32):
    touched = set()
    for lane in range(lanes):
        first = (offset + lane * stride) * itemsize
        touched.update(range(first // sector, (first + itemsize - 1) // sector + 1))
    return touched


def sectors_byte_reference(stride, offset=0, lanes=32):
    # 独立参考：枚举所有线程读取的每个 byte，不依赖区间端点公式。
    return {byte // 32 for lane in range(lanes)
            for byte in range(4 * (offset + lane * stride),
                              4 * (offset + lane * stride) + 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--offset', type=int, default=0)
    args = parser.parse_args()
    if args.stride < 0 or args.offset < 0:
        parser.error('本实验使用非负 stride 与 offset')
    base = np.arange(24, dtype=np.int32)
    x = base.reshape(2, 3, 4)
    assert tuple(s // x.itemsize for s in x.strides) == (12, 4, 1)
    for index in np.ndindex(x.shape):
        assert x[index] == base[address(index, (12, 4, 1))]
    y = x.transpose(0, 2, 1)
    for index in np.ndindex(y.shape):
        assert y[index] == base[address(index, (12, 1, 4))]
    sliced = x[:, 1:, ::2]
    for index in np.ndindex(sliced.shape):
        assert sliced[index] == base[address(index, (12, 4, 2), 4)]
    # 修改 view 可见；copy 独立。NumPy 的 .base 可能指向更早的祖先，
    # 因而用 shares_memory 检查真实共享，而不比较 .base is x。
    copied = np.ascontiguousarray(y)
    assert np.shares_memory(y, x) and not np.shares_memory(copied, x)
    assert np.array_equal(copied, y)
    original = int(x[0, 0, 1])
    y[0, 1, 0] = 999
    assert x[0, 0, 1] == 999 and copied[0, 1, 0] == original
    y[0, 1, 0] = original
    print('address / transpose / slice / alias checks: PASS')
    expected = {(1, 0): 4, (1, 1): 5, (2, 0): 8, (8, 0): 32, (0, 0): 1}
    for (stride, offset), count in expected.items():
        a = sectors_formula(stride, offset)
        b = sectors_byte_reference(stride, offset)
        assert a == b and len(a) == count
    # 穷举额外组合，覆盖跨 sector 与空 stride（广播）。
    for stride in range(17):
        for offset in range(33):
            assert sectors_formula(stride, offset) == sectors_byte_reference(stride, offset)
    got = len(sectors_formula(args.stride, args.offset))
    print(f'warp stride={args.stride}, offset={args.offset}: {got} sectors, {got * 32} sector-bytes')
    print('sector examples: aligned=4, offset1=5, stride2=8, stride8=32, broadcast=1')
    banks32 = [(lane * 32) % 32 for lane in range(32)]
    banks33 = [(lane * 33) % 32 for lane in range(32)]
    assert len(set(banks32)) == 1 and len(set(banks33)) == 32
    print('shared transpose pitch: 32 -> 1 bank; 33 -> 32 banks')
    b, t, h, d, bytes_per = 4, 2048, 32, 128, 2
    payload = b * t * h * d * bytes_per
    assert payload == 64 * 2**20
    print(f'BF16 copy: payload={payload // 2**20} MiB, read+write={2 * payload // 2**20} MiB')
    # 教学时间输入：一次复制 80 us；下游 stride kernel 140 us，连续 kernel 100 us。
    copy_us, strided_us, contiguous_us = 80, 140, 100
    assert copy_us + 2 * contiguous_us == 2 * strided_us
    print('copy break-even: 2 uses tie; 3 uses save 40 us (teaching timings)')
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
