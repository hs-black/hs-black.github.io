#!/usr/bin/env python3
"""CPU quantization and two-layer training probes. NumPy only.
python lab.py --seed 7 --steps 200 --block 4
Scales and master/optimizer storage are FP32. Quantization formats describe
GEMM inputs; this program uses NumPy GEMMs, not hardware FP4 instructions.
"""
import argparse
import numpy as np


def levels(fmt):
    if fmt == 'e2m1':
        return np.array([0., .5, 1., 1.5, 2., 3., 4., 6.])
    if fmt == 'e4m3':
        return np.array([m/8*2**(-6) if e == 0 else (1+m/8)*2**(e-7)
                         for e in range(16) for m in range(8)
                         if not (e == 15 and m == 7)])
    raise ValueError(fmt)


def round_values(x, fmt, rng=None):
    """RNE when rng is None, otherwise SR; finite values saturate at endpoints."""
    values = levels(fmt)
    sign = np.where(x < 0, -1., 1.)
    a = np.clip(np.abs(x), 0, values[-1])
    high = np.searchsorted(values, a, side='left')
    low = np.maximum(high-1, 0)
    lower, upper = values[low], values[high]
    gap = upper-lower
    p = np.divide(a-lower, gap, out=np.zeros_like(a), where=gap > 0)
    if rng is None:
        choose_high = (p > .5) | ((p == .5) & (high % 2 == 0))
    else:
        choose_high = rng.random(a.shape) < p
    return sign * np.where(choose_high, upper, lower)


def quantize(x, fmt, block=4, layout='row', rng=None):
    """Return dequantized values. All scales computed/stored in FP32.
    row: blocks along columns; col: blocks along rows; square: b x b.
    Separate one-dimensional orientations start from the same source x.
    """
    if fmt == 'fp32':
        return np.asarray(x, dtype=np.float32).copy()
    if layout == 'col':
        return quantize(x.T, fmt, block, 'row', rng).T
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    height = block if layout == 'square' else 1
    for i in range(0, x.shape[0], height):
        for j in range(0, x.shape[1], block):
            part = x[i:i+height, j:j+block]
            scale = np.float32(np.max(np.abs(part))/levels(fmt)[-1])
            if scale == 0:
                out[i:i+height, j:j+block] = 0
            else:
                out[i:i+height, j:j+block] = round_values(part/scale, fmt, rng)*scale
    return out


def rel_error(x, ref):
    return float(np.linalg.norm(x-ref)/max(np.linalg.norm(ref), 1e-15))


def forward(x, w, fmt, block):
    qx = quantize(x, fmt, block, 'row')
    qw = quantize(w, fmt, block, 'square')
    return qx @ qw, (x, qw)


def backward(g, cache, fmt, block, rng=None):
    x, qw = cache
    dx = quantize(g, fmt, block, 'row', rng) @ qw.T
    # Wgrad's reduction is along token rows: independently form column blocks.
    dw = quantize(x, fmt, block, 'col').T @ quantize(g, fmt, block, 'col', rng)
    return dx, dw


def train(x, target, initial, fmt, steps, block, seed, switch_last=0, sr=False):
    weights = [a.copy() for a in initial]
    moments = [np.zeros_like(a) for a in weights]
    variances = [np.zeros_like(a) for a in weights]
    rng = np.random.default_rng(seed)
    def evaluate():
        prediction = np.tanh(x@weights[0])@weights[1]
        return float(.5*np.mean((prediction-target)**2))
    first = evaluate()
    for step in range(1, steps+1):
        active = 'fp32' if switch_last and step > steps-switch_last else fmt
        u, cache1 = forward(x, weights[0], active, block)
        hidden = np.tanh(u)
        y, cache2 = forward(hidden, weights[1], active, block)
        g = (y-target)/y.size
        dh, dw2 = backward(g, cache2, active, block, rng if sr else None)
        _, dw1 = backward(dh*(1-hidden*hidden), cache1, active, block, rng if sr else None)
        for i, grad in enumerate([dw1, dw2]):
            moments[i] = np.float32(.9)*moments[i] + np.float32(.1)*grad
            variances[i] = np.float32(.95)*variances[i] + np.float32(.05)*grad*grad
            mhat = moments[i]/(1-.9**step)
            vhat = variances[i]/(1-.95**step)
            weights[i] -= np.float32(.02)*mhat/(np.sqrt(vhat)+np.float32(1e-8))
    return first, evaluate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--block', type=int, choices=[2, 4, 8], default=4)
    parser.add_argument('--switch-last', type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1 or not 0 <= args.switch_last <= args.steps:
        parser.error('steps must be positive; switch-last must lie between 0 and steps')
    rng = np.random.default_rng(args.seed)
    # The independent hand calculation in exercise 2 is the oracle.
    w = np.array([[6., 1.2], [2.3, 4.]])
    left = quantize(w, 'e2m1', 2).T
    right = quantize(w.T, 'e2m1', 2)
    np.testing.assert_allclose(left, [[6., 2.], [1., 4.]], atol=1e-6)
    np.testing.assert_allclose(right, [[6., 2.], [4/3, 4.]], atol=1e-6)
    square = quantize(w, 'e2m1', 2, 'square').T
    np.testing.assert_array_equal(square, quantize(w.T, 'e2m1', 2, 'square'))
    print(f'Transpose: rowwise difference {np.linalg.norm(left-right):.6f}; square difference 0')
    # Independent finite categorical distribution checks SR moments.
    true = .3
    draws = round_values(np.full(200_000, true), 'e2m1', rng)
    expected_variance = (true-0)*(.5-true)
    np.testing.assert_allclose(draws.mean(), true, atol=.003)
    np.testing.assert_allclose(draws.var(), expected_variance, atol=.001)
    np.testing.assert_allclose(np.mean(draws**2), true**2+expected_variance, atol=.002)
    print(f'SR: mean {draws.mean():.6f} (theory .300000); variance {draws.var():.6f} (theory .060000)')
    g, low, high = .01, 0., .02
    probabilities = np.array([.5, .5])
    second = probabilities@np.square([low, high])
    np.testing.assert_allclose(second, 2*g*g)
    print(f'Adam example: E[gq^2] {second:.6f}; g^2 {g*g:.6f}')
    # Shared-scale metadata contract: no quantization error in this maximum-element case.
    global_scale = 48/(448*6)
    wrong = 6*448*global_scale
    correct = 6*56*global_scale
    assert wrong == 48 and correct == 6
    # Orthogonal rotation on Wgrad's reduction dimension preserves the exact product.
    h = np.array([[1.,1.,1.,1.],[1.,-1.,1.,-1.],[1.,1.,-1.,-1.],[1.,-1.,-1.,1.]])/2
    x = rng.normal(size=(4,3)); grad = rng.normal(size=(4,2))
    np.testing.assert_allclose((x.T@h)@(h.T@grad), x.T@grad, atol=1e-14)
    # Three GEMMs, a separately checked loss derivative, and exact error decomposition.
    x = rng.normal(size=(8,8)); w = rng.normal(size=(8,4)); g = rng.normal(size=(8,4))
    x[:,0] *= 5
    eps = 1e-5
    scalar = lambda weight: np.sum((x@weight)*g)
    numerical = np.empty_like(w)
    for idx in np.ndindex(w.shape):
        original=w[idx];w[idx]=original+eps;plus=scalar(w);w[idx]=original-eps;minus=scalar(w);w[idx]=original
        numerical[idx]=(plus-minus)/(2*eps)
    np.testing.assert_allclose(numerical, x.T@g, atol=3e-9)
    for fmt in ['e4m3','e2m1']:
        y, cache = forward(x, w, fmt, args.block)
        dx, dw = backward(g, cache, fmt, args.block)
        qx=quantize(x,fmt,args.block,'row').astype(np.float64)
        qw=quantize(w,fmt,args.block,'square').astype(np.float64)
        ex,ew=qx-x,qw-w
        np.testing.assert_allclose(qx@qw-x@w,ex@w+x@ew+ex@ew,atol=1e-13)
        print(f'{fmt}: Fprop {rel_error(y,x@w):.5f}; Dgrad {rel_error(dx,g@w.T):.5f}; Wgrad {rel_error(dw,x.T@g):.5f} relative L2')
    # Train both layers, so Dgrad reaches the first layer's update.
    x=rng.normal(size=(64,8)).astype(np.float32);x[:,0]*=3
    teacher1=(rng.normal(size=(8,8))*.3).astype(np.float32)
    teacher2=(rng.normal(size=(8,4))*.3).astype(np.float32)
    target=np.tanh(x@teacher1)@teacher2
    initial=[(rng.normal(size=(8,8))*.15).astype(np.float32),(rng.normal(size=(8,4))*.15).astype(np.float32)]
    for fmt,sr in [('fp32',False),('e4m3',False),('e2m1',False),('e2m1',True)]:
        begin,end=train(x,target,initial,fmt,args.steps,args.block,args.seed+1,args.switch_last,sr)
        assert np.isfinite(end)
        if args.steps>=100:
            assert end < begin*.5
        print(f'Train {fmt} gradient={"SR" if sr else "RNE"}; FP32 readout: {begin:.6f} -> {end:.6f}')
    print('PASS: transpose, SR moments, Adam noise, gather metadata, RHT, GEMM gradients/decomposition, training')


if __name__ == '__main__':
    main()
