#!/usr/bin/env python3
"""CPU width-transfer lab. Requires NumPy; no downloaded model or data.

python3 lab.py             # independent identities + tiny coordinate check
python3 lab.py --sweep     # also compare three base learning rates, two seeds
The finite-width tanh regression experiment is a local diagnostic of this MLP.
"""
import argparse
import numpy as np


def init(width, din, base=16, mode="mup", seed=0):
    rng = np.random.default_rng(seed)
    ratio = width / base
    return [rng.normal(size=(width, din)) / np.sqrt(din),
            rng.normal(size=(width, width)) / np.sqrt(width),
            rng.normal(size=(1, width)) / np.sqrt(width * (ratio if mode == "mup" else 1))]


def forward(weights, x):
    w1, w2, w3 = weights
    h1 = np.tanh(x @ w1.T)
    h2 = np.tanh(h1 @ w2.T)
    return h2 @ w3.T, (h1, h2)


def loss_grad(weights, x, y):
    pred, (h1, h2) = forward(weights, x)
    residual = pred - y
    dy = residual / len(x)
    dz2 = (dy @ weights[2]) * (1 - h2**2)
    dz1 = (dz2 @ weights[1]) * (1 - h1**2)
    return float(np.mean(residual**2) / 2), [dz1.T @ x, dz2.T @ h1, dy.T @ h2]


def adam(weights, grads, state, rates, step, eps=1e-8):
    for w, g, (m, v), lr in zip(weights, grads, state, rates):
        m *= 0.9
        m += 0.1 * g
        v *= 0.999
        v += 0.001 * g**2
        w -= lr * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + eps)


def check_gradients():
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(7, 3)), rng.normal(size=(7, 1))
    weights = init(5, 3, base=5, seed=1)
    _, grads = loss_grad(weights, x, y)
    worst = 0.0
    for w, g in zip(weights, grads):
        for _ in range(8):
            idx = tuple(rng.integers(dim) for dim in w.shape)
            old = w[idx]
            w[idx] = old + 1e-5
            plus = loss_grad(weights, x, y)[0]
            w[idx] = old - 1e-5
            minus = loss_grad(weights, x, y)[0]
            w[idx] = old
            numerical = (plus - minus) / 2e-5
            worst = max(worst, abs(g[idx] - numerical))
            np.testing.assert_allclose(g[idx], numerical, rtol=2e-5, atol=2e-8)
    print(f"PASS central-difference gradients: max absolute error {worst:.2e}")


def check_readout_coordinates():
    # W_eff = V/r. Fixed features, independent optimization paths for 6 steps.
    rng = np.random.default_rng(9)
    r, eta, eps = 4.0, 0.02, 3e-3
    h, y = rng.normal(size=(11, 12)), rng.normal(size=(11, 1))
    initial = rng.normal(size=(1, 12)) / 12
    for optimizer in ("sgd", "adam"):
        effective, stored = initial.copy(), initial.copy() * r
        states = [[(np.zeros_like(initial), np.zeros_like(initial))] for _ in range(2)]
        for step in range(1, 7):
            ge = (h @ effective.T - y).T @ h / len(h)
            gs = (h @ (stored / r).T - y).T @ h / len(h) / r
            if optimizer == "sgd":
                effective -= (eta / r) * ge
                stored -= (eta * r) * gs
            else:
                # The epsilon scales with the gradient under coordinate change.
                adam([effective], [ge], states[0], [eta / r], step, eps)
                adam([stored], [gs], states[1], [eta], step, eps / r)
            np.testing.assert_allclose(effective, stored / r, rtol=2e-13, atol=2e-13)
        print(f"PASS effective/stored readout: {optimizer.upper()} matches through 6 steps")


def check_alignment_and_epsilon():
    eta, base = 0.01, 16
    for n in (16, 32, 64, 128):
        h = np.ones(n)
        g = 0.07 * h  # one-row, single-example gradient = delta * h
        sp_change = (-eta * g / np.abs(g)) @ h
        mup_change = (-(eta * base / n) * g / np.abs(g)) @ h
        np.testing.assert_allclose(sp_change, -eta * n)
        np.testing.assert_allclose(mup_change, -eta * base)
    print("PASS aligned hidden update: constant LR grows with width; LR/r stays -0.16")
    g, eps, c = 1e-7, 1e-8, 1e-3
    before = g / (abs(g) + eps)
    fixed = c * g / (abs(c * g) + eps)
    scaled = c * g / (abs(c * g) + c * eps)
    np.testing.assert_allclose(before, scaled, rtol=1e-14)
    assert fixed < before / 50
    print(f"PASS epsilon scaling: original={before:.6f}, fixed-epsilon={fixed:.6f}, scaled-epsilon={scaled:.6f}")


def data():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(64, 4))
    y = (np.sin(x[:, :1]) + 0.4 * x[:, 1:2] * x[:, 2:3])
    probe = rng.normal(size=(64, 4))
    target = np.sin(probe[:, :1]) + 0.4 * probe[:, 1:2] * probe[:, 2:3]
    return x, y, probe, target


def train(width, mode, eta=0.01, steps=50, seed=0):
    x, y, probe, target = data()
    weights = init(width, 4, seed=seed, mode=mode)
    _, initial_features = forward(weights, probe)
    ratio = width / 16
    rates = [eta, eta / ratio, eta / ratio] if mode == "mup" else [eta] * 3
    state = [(np.zeros_like(w), np.zeros_like(w)) for w in weights]
    start = loss_grad(weights, x, y)[0]
    for step in range(1, steps + 1):
        _, grads = loss_grad(weights, x, y)
        adam(weights, grads, state, rates, step)
    pred, features = forward(weights, probe)
    change = np.sqrt(np.mean((features[1] - initial_features[1])**2))
    result = (start, loss_grad(weights, x, y)[0], float(np.mean((pred-target)**2)/2), float(change))
    assert np.isfinite(result).all()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    check_gradients()
    check_readout_coordinates()
    check_alignment_and_epsilon()
    print("\nTiny tanh MLP: 5 Adam steps, eta=0.01, base width=16, seed=0")
    print("mode width train_loss_0 train_loss_5 probe_loss feature_change_RMS")
    for mode in ("sp", "mup"):
        for width in (16, 32, 64, 128):
            result = train(width, mode, steps=5)
            print(f"{mode:3s} {width:5d} " + " ".join(f"{value:.6f}" for value in result))
    if args.sweep:
        print("\n50-step sweep: probe loss averaged over seeds 0,1; one fixed dataset")
        print("mode width losses_at_eta_[0.003,0.01,0.03] selected_eta")
        rates = (0.003, 0.01, 0.03)
        for mode in ("sp", "mup"):
            for width in (16, 32, 64, 128):
                losses = [np.mean([train(width, mode, lr, seed=s)[2] for s in (0, 1)]) for lr in rates]
                print(f"{mode:3s} {width:5d} " + " ".join(f"{v:.6f}" for v in losses) + f" {rates[int(np.argmin(losses))]:.3f}")
        print("Interpret selected rates with their loss curves and seed variation; this is finite-width toy regression.")


if __name__ == "__main__":
    main()
