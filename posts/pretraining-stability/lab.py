#!/usr/bin/env python3
"""Numerical probes for normalization, entropy and updates. NumPy CPU only."""
import numpy as np


def rmsnorm(x, eps=1e-4):
    r = np.sqrt(np.mean(x*x) + eps)
    jac = np.eye(len(x))/r - np.outer(x, x)/(len(x)*r**3)
    return x/r, jac


def finite_jacobian(f, x, step=1e-6):
    columns = []
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = step
        columns.append((f(x+dx)-f(x-dx))/(2*step))
    return np.stack(columns, axis=1)


def stack(x, weights, placement):
    """RMSNorm + linear branches; final model norm omitted in this local probe."""
    jac = np.eye(len(x))
    for w in weights:
        if placement == "pre":
            normalized, norm_jac = rmsnorm(x)
            x = x + w @ normalized
            local = np.eye(len(x)) + w @ norm_jac
        elif placement == "post":
            x, norm_jac = rmsnorm(x + w @ x)
            local = norm_jac @ (np.eye(len(x))+w)
        else:
            raise ValueError(placement)
        jac = local @ jac
    return x, jac


def entropy(scale, scores):
    logits = scale*scores
    logits -= logits.max()
    logp = logits - np.log(np.exp(logits).sum())
    p = np.exp(logp)
    h = -np.dot(p, logp)
    derivative = -scale * np.dot(p, (scores-np.dot(p, scores))**2)
    return h, derivative


def clip(g, threshold):
    return g * min(1., threshold / max(np.linalg.norm(g), 1e-30))


def main():
    x = np.array([.3, -.9, 1.2, .6])
    _, analytic = rmsnorm(x)
    numerical = finite_jacobian(lambda z: rmsnorm(z)[0], x)
    np.testing.assert_allclose(analytic, numerical, atol=3e-10)
    rng = np.random.default_rng(13)
    weights = rng.normal(size=(6, 4, 4)) * .15
    for mode in ["pre", "post"]:
        out, jac = stack(x, weights, mode)
        numeric = finite_jacobian(lambda z: stack(z, weights, mode)[0], x)
        np.testing.assert_allclose(jac, numeric, atol=3e-9)
        upstream = np.array([1., -.5, .2, .7])
        np.testing.assert_allclose(jac.T @ upstream, numeric.T @ upstream, atol=3e-9)
        print(f"{mode}: output RMS {np.sqrt(np.mean(out*out)):.4f}, input-gradient norm {np.linalg.norm(jac.T@upstream):.4f}")
    scores = np.array([1., 0., -1.])
    for scale in [0., .1, 1., 3., 8.]:
        h, dh = entropy(scale, scores)
        numeric = (entropy(scale+1e-5, scores)[0]-entropy(scale-1e-5, scores)[0])/2e-5
        np.testing.assert_allclose(dh, numeric, atol=2e-10)
        assert dh <= 0
    print(f"Attention entropy: scale 1 -> {entropy(1., scores)[0]:.4f}; scale 8 -> {entropy(8., scores)[0]:.4f} nats")
    # Exact SVD isolates the reparameterization algebra; production uses power iteration.
    w = rng.normal(size=(4, 4))
    gamma = 1.3
    effective = gamma*w/np.linalg.norm(w, 2)
    scaled = gamma*(9*w)/np.linalg.norm(9*w, 2)
    np.testing.assert_allclose(effective, scaled, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(effective, 2), gamma)
    g, loss_scale, threshold = np.array([3., 4.]), 128., 1.
    correct = clip(loss_scale*g/loss_scale, threshold)
    wrong = clip(loss_scale*g, threshold)/loss_scale
    np.testing.assert_allclose(np.linalg.norm(correct), 1.)
    np.testing.assert_allclose(np.linalg.norm(wrong), 1/128)
    # Adam's first step approximately cancels a uniform gradient rescaling.
    first_adam = lambda grad: grad/(np.abs(grad)+1e-8)
    np.testing.assert_allclose(first_adam(g), first_adam(clip(g, 1.)), atol=2e-8)
    # A deterministic growth counterexample: identity residual path alone gives no bound.
    growth = (1+.02)**100
    assert growth > 7
    print(f"100 scalar residual Jacobians of 1.02: gain {growth:.4f}")
    print(f"Unscale then clip: norm {np.linalg.norm(correct):.6f}; reversed: {np.linalg.norm(wrong):.6f}")
    print("PASS: RMSNorm/stack finite differences, entropy derivative, spectral scaling, clipping order, Adam probe")


if __name__ == "__main__":
    main()
