#!/usr/bin/env python3
"""NumPy optimizer arithmetic: python3 lab.py. Uses FP64, no model downloads."""
import numpy as np


def gd(start, lr, steps):
    x = np.array(start, dtype=np.float64)
    for _ in range(steps):
        x -= lr * np.array([x[0], 25 * x[1]])
    return x


def main():
    x = np.array([1.2, -.7]); eps = 1e-6
    def loss(q): return .5 * (q[0] ** 2 + 25 * q[1] ** 2)
    analytic = np.array([x[0], 25*x[1]])
    numeric = []
    for i in range(2):
        delta = np.eye(2)[i] * eps
        numeric.append((loss(x+delta)-loss(x-delta))/(2*eps))
    np.testing.assert_allclose(numeric, analytic, rtol=1e-8)
    for lr in [.05, .075, .08, .09]:
        result = gd([1., 1.], lr, 100)
        exact = np.array([(1-lr)**100, (1-25*lr)**100])
        np.testing.assert_allclose(result, exact, rtol=1e-11, atol=1e-12)
    assert abs(gd([1.,1.], .075,100)[1]) < 1
    assert abs(gd([1.,1.], .09,100)[1]) > 1
    print('PASS: gradient finite difference and GD stability')
    m = .1*2; v=.001*4
    direction=(m/(1-.9))/np.sqrt(v/(1-.999))
    np.testing.assert_allclose(direction,1.,rtol=1e-12)
    theta=1.
    for _ in range(1000): theta *= 1-.01*.1
    np.testing.assert_allclose(theta,.999**1000,rtol=1e-12)
    print(f'PASS: Adam bias correction; AdamW decay after 1000 steps={theta:.6f}')
    rng=np.random.default_rng(7)
    for shape in [(4,8),(8,4),(4,4)]:
        m=rng.normal(size=shape)
        u,_,vh=np.linalg.svd(m,full_matrices=False)
        polar=u@vh
        np.testing.assert_allclose(np.sqrt(np.mean(polar**2)),1/np.sqrt(max(shape)),rtol=1e-12)
        scaled=polar*.2*np.sqrt(max(shape))
        np.testing.assert_allclose(np.sqrt(np.mean(scaled**2)),.2,rtol=1e-12)
    print('PASS: full-rank polar-factor RMS and shape scaling')
    print('All optimizer arithmetic checks PASS')


if __name__=='__main__': main()
