#!/usr/bin/env python3
"""Small, reproducible pruning experiments. Run: python3 lab.py (NumPy only)."""
import itertools
import numpy as np

def close(a, b, name):
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10, err_msg=name)
    print("PASS:", name)

def all_masks(size, kept):
    for ids in itertools.combinations(range(size), kept):
        mask = np.zeros(size)
        mask[list(ids)] = 1
        yield mask

def main():
    # Orthonormal rows give a finite dataset with exactly diagonal covariance.
    x = np.array([[1, 1, -1, -1], [1, -1, 1, -1.]], dtype=float)
    x[0] *= 8
    w = np.array([[.1, .6]])
    deletion_errors = []
    for j in range(2):
        pruned = w.copy()
        pruned[0, j] = 0
        deletion_errors.append(np.mean((w @ x - pruned @ x)**2))
    close(deletion_errors, [.64, .36], "Wanda scores squared equal isolated deletion MSE")
    w = np.array([[.9, .8, .7, .6], [.5, .4, .3, .2]])
    rms = np.array([1, 3, .5, 2.])
    # An explicit, independent forward reference on an 16-point sign dataset.
    x = np.array(list(itertools.product([-1., 1.], repeat=4))).T * rms[:, None]
    score = np.abs(w) * rms
    mask = np.zeros(8)
    mask[np.argsort(-score.ravel(), kind="stable")[:4]] = 1
    measured = lambda m: np.mean(np.sum((w @ x - (w*m.reshape(2, 4)) @ x)**2, axis=0))
    close(measured(mask), min(measured(m) for m in all_masks(8, 4)),
          "global Wanda minimizes diagonal-covariance reconstruction over 70 masks")
    masks24 = [np.concatenate([a, b]) for a in all_masks(4, 2) for b in all_masks(4, 2)]
    mask24 = np.zeros((2, 4))
    for row in range(2):
        mask24[row, np.argsort(-score[row], kind="stable")[:2]] = 1
    close(measured(mask24), min(measured(m) for m in masks24),
          "local Wanda minimizes 2:4 reconstruction over 36 masks")

    rng = np.random.default_rng(14)
    x = rng.normal(size=(5, 37))
    x[1] = .7*x[0] + .4*x[1]
    w = rng.normal(size=5)
    h = x @ x.T / x.shape[1]
    j = 2
    inv_col = np.linalg.solve(h, np.eye(5)[:, j])
    delta = -w[j]*inv_col/inv_col[j]
    obs = w + delta
    remaining = [k for k in range(5) if k != j]
    # Independent constrained least-squares solve from raw input/output, without H.
    ref = np.zeros(5)
    ref[remaining] = np.linalg.lstsq(x[remaining].T, w@x, rcond=None)[0]
    close(obs, ref, "OBS compensation equals independent constrained least squares")
    close(np.mean(((w-obs)@x)**2), w[j]**2/inv_col[j],
          "OBS cost matches measured reconstruction error")
    for rho in [-.8, 0., .8]:
        x1 = np.array([1., -1, 1, -1])
        z = np.array([1., 1, -1, -1])
        x2 = rho*x1 + np.sqrt(1-rho*rho)*z
        close(np.mean((x1+x2-(1+rho)*x2)**2), 1-rho*rho,
              f"correlation compensation rho={rho}")

    n = 4096
    dense = n*n*2
    csr = n*n//2*(2+4)+(n+1)*4
    packed24 = n*n//4*(2*2)+n*n//4*4//8
    close([dense, csr, packed24], [33554432, 50348036, 18874368],
          "dense / CSR / 2:4 storage bytes")
    x = np.array([1., -1.])
    w = np.ones(2)
    hard_losses = [(np.dot(w*m, x))**2 for m in all_masks(2, 1)]
    soft_loss = np.dot(.5*w, x)**2
    close(hard_losses, [1., 1.], "both hard 1:2 masks have unit error")
    close(soft_loss, 0., "soft mask can conceal hard-export error")
    print("All checks passed. These are CPU algebra experiments; no GPU timing is inferred.")

if __name__ == "__main__":
    main()
