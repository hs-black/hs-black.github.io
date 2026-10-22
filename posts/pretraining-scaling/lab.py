#!/usr/bin/env python3
"""CPU scaling-law laboratory. Python 3 + NumPy; no model/data download.

All observations here are synthetic. N is measured in 1B parameters and
D in 20B tokens. We fit squared loss residuals with a grid over exponents
and a conditional linear least-squares solve, then test held-out scales.
The research papers use other fitting objectives (including log-Huber).
"""

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Law:
    e: float = 1.5
    a: float = 0.8
    b: float = 0.8
    alpha: float = 0.3
    beta: float = 0.3

    def loss(self, n, d):
        return self.e + self.a * np.asarray(n) ** (-self.alpha) + self.b * np.asarray(d) ** (-self.beta)

    def optimum(self, c):
        n = (self.alpha * self.a / (self.beta * self.b) * c**self.beta) ** (1 / (self.alpha + self.beta))
        return n, c / n


def fit(n, d, loss):
    """Profile out E,A,B at every alpha,beta; keep physically positive fits."""
    best = None
    for alpha in np.linspace(0.15, 0.45, 13):
        for beta in np.linspace(0.15, 0.45, 13):
            design = np.column_stack([np.ones_like(n), n**-alpha, d**-beta])
            coeff, _, rank, _ = np.linalg.lstsq(design, loss, rcond=None)
            if rank < 3 or np.any(coeff <= 0):
                continue
            mse = np.mean((design @ coeff - loss) ** 2)
            if best is None or mse < best[0]:
                best = (mse, Law(*coeff, alpha, beta))
    if best is None:
        raise ValueError("No positive, full-rank fit; expand the experimental grid.")
    return best[1]


def effective_data(d, unique=1.0, repeat_scale=3.0):
    """Units match D: first pass counts fully; later marginal value decays."""
    d = np.asarray(d)
    extra = np.maximum(d / unique - 1, 0)
    repeated = unique * (1 + repeat_scale * (-np.expm1(-extra / repeat_scale)))
    return np.where(d <= unique, d, repeated)


def tokens_for_loss(law, n, target):
    remaining = target - law.e - law.a * np.asarray(n) ** (-law.alpha)
    if np.any(remaining <= 0):
        raise ValueError("Target is infeasible at this parameter count.")
    return (law.b / remaining) ** (1 / law.beta)


def deployment_optimum(law, target, inference_billion):
    """Minimize FLOPs: train=1.2e20*n*d; infer=2e18*n*I_B."""
    lower = (law.a / (target - law.e)) ** (1 / law.alpha)
    grid = np.geomspace(lower * 1.0001, lower * 1000, 100_001)
    d = tokens_for_loss(law, grid, target)
    cost = 1.2 * grid * d + 0.02 * grid * inference_billion  # 1e20 FLOPs
    index = np.argmin(cost)
    assert 0 < index < len(grid) - 1, "Expand the search range."
    return grid[index], d[index], cost[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise", type=float, default=0.005, help="Synthetic loss noise, nats/token")
    parser.add_argument("--bootstrap", type=int, default=80)
    args = parser.parse_args()
    if args.noise < 0 or args.bootstrap < 10:
        parser.error("noise must be nonnegative; bootstrap must be at least 10")
    rng = np.random.default_rng(args.seed)
    truth = Law()
    n, d = np.meshgrid([0.25, 0.5, 1.0, 2.0], [0.25, 0.5, 1.0, 2.0])
    n, d = n.ravel(), d.ravel()

    # Independent checks: recover a known generating law, compare the
    # constrained closed form with a dense search, and finite-difference D_eff.
    exact = fit(n, d, truth.loss(n, d))
    np.testing.assert_allclose([exact.e, exact.a, exact.b, exact.alpha, exact.beta], [1.5, .8, .8, .3, .3], atol=1e-10)
    general = Law(alpha=.25, beta=.4)
    for c in [0.5, 1, 4, 16]:
        expected_n, expected_d = general.optimum(c)
        grid = np.geomspace(expected_n / 4, expected_n * 4, 20_001)
        observed_n = grid[np.argmin(general.loss(grid, c / grid))]
        np.testing.assert_allclose(observed_n, expected_n, rtol=1e-4)
        np.testing.assert_allclose(expected_n * expected_d, c)
    h = 1e-5
    finite_difference = (effective_data(4 + h) - effective_data(4 - h)) / (2 * h)
    np.testing.assert_allclose(finite_difference, np.exp(-1), rtol=1e-8)
    np.testing.assert_allclose(effective_data(1000), 4)
    print("Independent checks: exact fit, constrained optimum, repeat derivative/limit PASS")

    observed = truth.loss(n, d) + rng.normal(0, args.noise, len(n))
    fitted = fit(n, d, observed)
    # Every validation point is outside the fit rectangle along at least one
    # axis. No checkpoints from a fit run enter this validation set.
    hn, hd = np.meshgrid([0.5, 1, 2, 4, 8], [0.5, 1, 2, 4, 8])
    outside = (hn > 2) | (hd > 2)
    hn, hd = hn[outside], hd[outside]
    residual = fitted.loss(hn, hd) - truth.loss(hn, hd)
    print(f"Synthetic fit: E={fitted.e:.4f} A={fitted.a:.4f} B={fitted.b:.4f} alpha={fitted.alpha:.3f} beta={fitted.beta:.3f}")
    print(f"Held-out scale RMSE: {np.sqrt(np.mean(residual**2)):.6f} nats/token; mean residual={np.mean(residual):+.6f}")
    print(f"Predicted c=16 optimum: N={fitted.optimum(16)[0]:.3f}B; D={20*fitted.optimum(16)[1]:.3f}B tokens")

    predictions = []
    for _ in range(args.bootstrap):
        sample = rng.integers(0, len(n), len(n))
        try:
            boot = fit(n[sample], d[sample], observed[sample])
        except ValueError:
            continue
        predictions.append(boot.optimum(16)[0])
    if len(predictions) < 10:
        raise RuntimeError("Too few identifiable bootstrap samples")
    lo, hi = np.quantile(predictions, [0.1, 0.9])
    print(f"Run-bootstrap 10%-90% N interval at c=16: [{lo:.3f}, {hi:.3f}]B ({len(predictions)} fits)")

    # Distribution shift: the fitting grid had fresh data, while validation
    # repeats a fixed 80B-token pool. The same model form now misses the cause.
    long_d = np.array([8., 16., 32.])
    actual = truth.loss(1, effective_data(long_d, unique=4))
    naive = truth.loss(1, long_d)
    print("Finite-data extrapolation (unique=80B tokens, R*=3):")
    for tokens, predicted, heldout in zip(20 * long_d, naive, actual):
        print(f"  processed={tokens:.0f}B: fresh-data prediction={predicted:.4f}; repeat-aware loss={heldout:.4f}; optimism={heldout-predicted:.4f}")

    print("Fixed target loss=3.1; training + inference FLOPs:")
    previous_n = float("inf")
    for demand in [0, 20, 100, 200]:
        best_n, best_d, cost = deployment_optimum(truth, 3.1, demand)
        np.testing.assert_allclose(truth.loss(best_n, best_d), 3.1, atol=1e-12)
        assert best_n <= previous_n
        previous_n = best_n
        if demand == 0:
            np.testing.assert_allclose(best_n, 1, rtol=1e-4)
        print(f"  inference={demand:3d}B tokens: N={best_n:.3f}B D={20*best_d:.2f}B total={cost:.3f}e20 FLOPs")
    target = truth.loss(2, 2)
    small_d = tokens_for_loss(truth, 1, target)
    extra_training = 6 * 1e9 * small_d * 20e9 - 6 * 2e9 * 40e9
    break_even = extra_training / (2 * (2e9 - 1e9))
    large_cost = 6 * 2e9 * 40e9 + 2 * 2e9 * break_even
    small_cost = 6 * 1e9 * small_d * 20e9 + 2 * 1e9 * break_even
    np.testing.assert_allclose(large_cost, small_cost, rtol=1e-12)
    print(f"Exercise 3: small model trains on {20*small_d:.4f}B tokens; extra train={extra_training:.6e} FLOPs; break-even={break_even/1e9:.4f}B inference tokens")
    print("All checks PASS. Numbers are synthetic predictions, not GPU measurements.")


if __name__ == "__main__":
    main()
