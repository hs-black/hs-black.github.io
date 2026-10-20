#!/usr/bin/env python3
"""LoRA 数值实验：python3 lab.py [--rank 1|2|4] [--steps 3000].
依赖 NumPy。float64 用于核对梯度，不模拟 GPU 速度或真实训练显存。
NF4 采用 bitsandbytes 的公开码本，演示分块/打包/重建；不实现 double quant。
https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py
"""
import argparse
from dataclasses import dataclass
import numpy as np

NF4 = np.array([-1., -.6961928009986877, -.5250730514526367,
    -.39491748809814453, -.28444138169288635, -.18477343022823334,
    -.09105003625154495, 0., .07958029955625534, .16093020141124725,
    .24611230194568634, .33791524171829224, .44070982933044434,
    .5626170039176941, .7229568362236023, 1.])

@dataclass
class Quantized:
    packed: np.ndarray
    scales: np.ndarray
    shape: tuple
    block: int
    size: int

def quantize_nf4(w, block=64):
    """全零块用 scale=0；最后一块允许不足 block；每字节存两个索引。"""
    flat = np.asarray(w).ravel()
    if block < 1: raise ValueError('block must be positive')
    codes = np.empty(flat.size, dtype=np.uint8)
    scales = np.empty((flat.size + block - 1) // block, dtype=np.float32)
    for j, start in enumerate(range(0, flat.size, block)):
        values = flat[start:start+block]
        scale = np.max(np.abs(values))
        scales[j] = scale
        normalized = values / scales[j] if scale else np.zeros_like(values)
        codes[start:start+values.size] = np.argmin(abs(normalized[:, None]-NF4), axis=1)
    # padding 只用于存储，重建时按原元素数截断。
    if codes.size % 2: codes = np.append(codes, np.uint8(7))
    packed = codes[0::2] | (codes[1::2] << 4)
    return Quantized(packed, scales, w.shape, block, flat.size)

def dequantize(q):
    codes = np.empty(q.packed.size * 2, dtype=np.uint8)
    codes[0::2] = q.packed & 15
    codes[1::2] = q.packed >> 4
    flat = NF4[codes[:q.size]]
    return (flat * np.repeat(q.scales, q.block)[:q.size]).reshape(q.shape)

def forward(x, w, a, b, scale):
    return x @ w.T + scale * (x @ a.T) @ b.T

def loss_and_grads(x, y, w, a, b, scale):
    z = x @ a.T
    residual = forward(x, w, a, b, scale) - y
    g = residual / len(x)
    return .5*np.sum(residual**2)/len(x), scale*(g@b).T@x, scale*g.T@z

def finite_difference(x, y, w, a, b, scale):
    _, da, db = loss_and_grads(x, y, w, a, b, scale)
    worst = 0.
    for param, expected in ((a, da), (b, db)):
        for idx in np.ndindex(param.shape):
            old = param[idx]; eps = 1e-6
            param[idx] = old + eps
            plus = .5*np.sum((forward(x,w,a,b,scale)-y)**2)/len(x)
            param[idx] = old - eps
            minus = .5*np.sum((forward(x,w,a,b,scale)-y)**2)/len(x)
            param[idx] = old
            worst = max(worst, abs((plus-minus)/(2*eps)-expected[idx]))
    assert worst < 1e-8, worst
    return worst

def train(x, y, xt, yt, w, rank, steps):
    rng = np.random.default_rng(23)
    a = rng.normal(0, .1, (rank, w.shape[1]))
    b = np.zeros((w.shape[0], rank)); scale = 1.  # alpha = rank
    frozen = w.copy()
    before, da, db = loss_and_grads(x, y, w, a, b, scale)
    assert np.max(abs(da)) == 0 and np.linalg.norm(db) > 0
    for _ in range(steps):
        _, da, db = loss_and_grads(x, y, w, a, b, scale)
        a -= .1*da; b -= .1*db
    after, _, _ = loss_and_grads(x, y, w, a, b, scale)
    test = .5*np.sum((forward(xt,w,a,b,scale)-yt)**2)/len(xt)
    assert np.array_equal(w, frozen), 'base weights changed'
    assert after < before, (before, after)
    merged = w + scale*b@a
    merge_error = np.max(abs(forward(xt,w,a,b,scale)-xt@merged.T))
    assert merge_error < 1e-12
    return before, after, test, merge_error

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()
    if not 1 <= args.rank <= 8 or args.steps < 1: parser.error('rank: 1..8; steps >= 1')
    rng = np.random.default_rng(7)
    w = rng.normal(0, .3, (6,8))
    # 已知 rank=2 的任务变化；训练集和测试集独立。
    delta = rng.normal(0,.2,(6,2)) @ rng.normal(0,.2,(2,8))
    x = rng.normal(size=(96,8)); xt = rng.normal(size=(64,8))
    y = x@(w+delta).T; yt = xt@(w+delta).T
    a = rng.normal(0,.1,(2,8)); b = rng.normal(0,.1,(6,2))
    err = finite_difference(x[:5], y[:5], w, a, b, 1.5)
    print(f'gradient finite difference: PASS (max abs error {err:.2e})')
    q = quantize_nf4(w, block=16); wq = dequantize(q)
    # 独立逐元素检查码本距离，包含零块和奇数长度。
    for value in (np.zeros((3,3)), np.array([[0., -.1, .2, -.8, 1.]]), w):
        qt = quantize_nf4(value, block=16); dq = dequantize(qt).ravel()
        for i, raw in enumerate(value.ravel()):
            s = float(qt.scales[i//16]); candidates = NF4*s
            assert abs(dq[i]-raw) <= np.min(abs(candidates-raw)) + 1e-12
        assert qt.packed.nbytes == (value.size+1)//2
    print(f'NF4 pack / nearest-code / zero-block / odd-tail: PASS')
    print(f'NF4 storage: codes={q.packed.nbytes} B, scales={q.scales.nbytes} B (block=16, no double quant)')
    for name, base in [('float base', w), ('NF4 base', wq)]:
        before, after, test, merge_err = train(x,y,xt,yt,base,args.rank,args.steps)
        print(f'{name}, rank={args.rank}: train {before:.6f} -> {after:.6f}; test {test:.6f}; merge error {merge_err:.2e}')
    # 账本独立于运行时 float64；明确使用 BF16 + FP32 master/Adam 假设。
    params = 32*2*16*(4096+4096)
    assert params == 8_388_608
    assert params*2/2**20 == 16 and params*16/2**20 == 128
    print('32 layers x Q,V x rank16: adapter weights=16 MiB; trainable state=128 MiB')
    print('PASS: frozen base, gradient, merged inference, quantization and memory accounting')

if __name__ == '__main__': main()
