#!/usr/bin/env python3
"""Controlled CPU experiments for residual geometry and layer interventions."""
import argparse
import numpy as np


def center(x):
    return x - x.mean(axis=0, keepdims=True)


def cka(x, y):
    x, y = center(x), center(y)
    return np.linalg.norm(x.T @ y, 'fro')**2 / (np.linalg.norm(x.T @ x, 'fro') *
                                               np.linalg.norm(y.T @ y, 'fro'))


def similarity(states):
    return np.array([[cka(x, y) for y in states] for x in states])


def effective_depth(matrix):
    layers = len(matrix)
    rho = np.array([np.diag(matrix, k).mean() for k in range(1, layers)])
    denominator = 1 + 2*np.sum((1-np.arange(1, layers)/layers)*rho)
    return layers / denominator


def construction(layers=6, width=3, gamma=0., correlation=0.):
    """Exact orthogonal sample subspaces, including initial and shared components."""
    rng = np.random.default_rng(17)
    columns = (layers+2)*width
    raw = center(rng.normal(size=(columns+8, columns)))
    orthogonal, _ = np.linalg.qr(raw, mode='reduced')
    chunks = np.split(orthogonal, layers+2, axis=1)
    initial, common, *independent = chunks
    state = np.sqrt(gamma)*initial
    states, updates = [], []
    for new in independent:
        update = np.sqrt(correlation)*common + np.sqrt(1-correlation)*new
        state = state + update
        updates.append(update)
        states.append(state.copy())
    return states, updates


def analytic_cka(layers, gamma, correlation):
    i = np.arange(1, layers+1, dtype=float)
    cross = gamma + correlation*np.outer(i, i) + (1-correlation)*np.minimum.outer(i, i)
    variance = gamma + correlation*i*i + (1-correlation)*i
    return cross**2 / np.outer(variance, variance)


def main():
    for layers in (2, 6, 24):
        states, updates = construction(layers)
        measured = similarity(states)
        np.testing.assert_allclose(measured, analytic_cka(layers, 0., 0.), atol=1e-12)
        depth = effective_depth(measured)
        np.testing.assert_allclose(depth, 2*layers/(layers+1), atol=1e-12)
        np.testing.assert_allclose(effective_depth(similarity(updates)), layers, atol=1e-12)
        print('PASS orthogonal updates: L=%d accumulated D=%.6f; update D=%.6f' %
              (layers, depth, effective_depth(similarity(updates))))

    for gamma, corr in ((4., 0.), (0., .5), (4., .5), (0., 1.)):
        states, updates = construction(gamma=gamma, correlation=corr)
        np.testing.assert_allclose(similarity(states), analytic_cka(6, gamma, corr), atol=1e-12)
        print('PASS carry/correlation: gamma=%.1f c=%.1f state D=%.6f update D=%.6f' %
              (gamma, corr, effective_depth(similarity(states)), effective_depth(similarity(updates))))

    rng = np.random.default_rng(20261022)
    x, y = rng.normal(size=(40, 5)), rng.normal(size=(40, 5))
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    np.testing.assert_allclose(cka(x, y), cka(7*x @ q, y), atol=1e-12)
    # A sample-space Gram calculation independently checks the feature-space implementation.
    gx, gy = center(x) @ center(x).T, center(y) @ center(y).T
    gram_cka = np.sum(gx*gy)/(np.linalg.norm(gx, 'fro')*np.linalg.norm(gy, 'fro'))
    np.testing.assert_allclose(cka(x, y), gram_cka, atol=1e-12)
    print('PASS CKA invariance and feature/sample-space agreement')

    # Online sufficient statistics use the global mean, including between-batch covariance.
    x[:20] += 4
    y[:20] += 2
    count = 0
    sx, sy, cross = np.zeros(5), np.zeros(5), np.zeros((5, 5))
    locally_centered = np.zeros_like(cross)
    for xb, yb in zip(np.array_split(x, 2), np.array_split(y, 2)):
        count += len(xb)
        sx += xb.sum(axis=0)
        sy += yb.sum(axis=0)
        cross += xb.T @ yb
        locally_centered += center(xb).T @ center(yb)
    global_cross = cross - np.outer(sx, sy)/count
    np.testing.assert_allclose(global_cross, center(x).T @ center(y), atol=1e-10)
    assert np.linalg.norm(global_cross-locally_centered) > 1
    print('PASS global centering: batch statistics recover full covariance')

    h, update, weight = np.array([10., 0.]), np.array([0., .1]), np.array([0., 20.])
    cosine = h @ (h+update)/(np.linalg.norm(h)*np.linalg.norm(h+update))
    full_loss = np.logaddexp(0, -weight @ (h+update))
    skipped_loss = np.logaddexp(0, -weight @ h)
    np.testing.assert_allclose(full_loss, .1269280110429725, atol=1e-12)
    np.testing.assert_allclose(skipped_loss-full_loss, .5662191695169727, atol=1e-12)
    print('PASS feature intervention: cosine %.6f; loss increase %.6f nats' %
          (cosine, skipped_loss-full_loss))

    h0, u = 2., 3.
    def network(skip_first, skip_second):
        state = h0 if skip_first else h0+u
        return state if skip_second else state-state
    baseline = network(False, False)
    delta1 = network(True, False)-baseline
    delta2 = network(False, True)-baseline
    joint = network(True, True)-baseline
    assert delta1 == 0 and delta2 == 5 and joint == 2 and delta1+delta2 != joint
    p, q = np.array([.9, .1]), np.array([.6, .4])
    kl = np.sum(p*np.log(p/q))
    np.testing.assert_allclose(kl, .22628916118535888, atol=1e-12)
    print('PASS joint interventions and prediction KL = %.6f nats' % kl)
    print('All numerical checks passed.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gamma', type=float, default=0., help='initial-state Gram scale')
    parser.add_argument('--correlation', type=float, default=0., help='update cross-Gram coefficient, from 0 to 1')
    args = parser.parse_args()
    if args.gamma < 0 or not 0 <= args.correlation <= 1:
        parser.error('gamma must be nonnegative and correlation must be in [0, 1]')
    states, updates = construction(gamma=args.gamma, correlation=args.correlation)
    print('experiment: gamma=%.2f c=%.2f accumulated D=%.6f; update D=%.6f' %
          (args.gamma, args.correlation, effective_depth(similarity(states)), effective_depth(similarity(updates))))


if __name__ == '__main__':
    main()
