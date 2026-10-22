#!/usr/bin/env python3
"""Model-merging contracts on CPU. Python 3 + NumPy, no downloads.

The parameter vectors and regression data are synthetic. This experiment
checks coordinate alignment, TIES arithmetic, DARE moments, vocabulary
alignment, and validation-only model selection with a separate test set.
"""
import itertools
import numpy as np


def network(x, incoming, outgoing):
    return np.maximum(np.asarray(x)[:, None] * incoming[None, :], 0) @ outgoing


def trim_rows(delta, keep):
    out = np.zeros_like(delta)
    for i, row in enumerate(delta):
        indices = np.argsort(-np.abs(row), kind="stable")[:keep]
        out[i, indices] = row[indices]
    return out


def ties(delta, keep):
    """Magnitude sign election; exactly tied coordinates have zero update.

    This tutorial's explicit tie rule differs from the original repository's
    global-majority sign fallback. All non-tied coordinates use the same mean.
    """
    trimmed = trim_rows(delta, keep)
    sign = np.sign(trimmed.sum(axis=0))
    selected = (np.sign(trimmed) == sign[None, :]) & (trimmed != 0)
    count = selected.sum(axis=0)
    merged = np.divide((trimmed * selected).sum(axis=0), count,
                       out=np.zeros(delta.shape[1]), where=count > 0)
    return merged, trimmed, sign


def mse(x, y, theta):
    return np.mean((x @ theta - y) ** 2)


def greedy_soup(models, xval, yval):
    order = sorted(range(len(models)), key=lambda i: mse(xval, yval, models[i]))
    selected = [order[0]]
    current = models[order[0]].copy()
    for i in order[1:]:
        proposal = (len(selected) * current + models[i]) / (len(selected) + 1)
        if mse(xval, yval, proposal) <= mse(xval, yval, current):
            current = proposal
            selected.append(i)
    return current, selected


def main():
    # Same function, permuted hidden units; aligning both incident matrices
    # restores identical coordinates before averaging.
    x = np.linspace(-1, 1, 2001)
    wa, va = np.array([1., -1.]), np.ones(2)
    wb, vb = wa[::-1].copy(), va[::-1].copy()
    np.testing.assert_allclose(network(x, wa, va), np.abs(x))
    np.testing.assert_allclose(network(x, wb, vb), np.abs(x))
    mid = network(x, (wa + wb) / 2, (va + vb) / 2)
    np.testing.assert_allclose(mid, 0)
    best_perm = min(itertools.permutations(range(2)),
                    key=lambda p: np.sum((wa - wb[list(p)]) ** 2)
                    + np.sum((va - vb[list(p)]) ** 2))
    aligned = network(x, (wa + wb[list(best_perm)]) / 2,
                      (va + vb[list(best_perm)]) / 2)
    np.testing.assert_allclose(aligned, np.abs(x))
    print(f"Permutation: direct midpoint MSE={np.mean((mid-np.abs(x))**2):.6f}; aligned MSE={np.mean((aligned-np.abs(x))**2):.6f}")

    delta = np.array([[.8, -.6, .05, .02], [.7, .9, -.04, .01], [-2., .4, .03, -.06]])
    merged, trimmed, sign = ties(delta, keep=2)
    np.testing.assert_allclose(merged, [-2., .65, 0., 0.])
    np.testing.assert_allclose(sign, [-1, 1, 0, 0])
    np.testing.assert_allclose(ties(delta[[2, 0, 1]], 2)[0], merged)
    theta0 = np.array([3., 2., 1., -1.])
    weights = np.array([.2, .3, .5])
    np.testing.assert_allclose(weights @ (delta + theta0), theta0 + weights @ delta)
    print("TIES (keep 2 of 4):", merged, "; merged theta:", theta0 + merged)

    # DARE: estimate first and second moments independently via sampling.
    rng = np.random.default_rng(23)
    q, width, increment, trials = .25, 16, .2, 100_000
    masks = rng.random((trials, width)) < q
    outputs = increment * masks.mean(axis=1) / q
    expected_var = increment**2 * (1-q) / (q*width)
    stderr = np.sqrt(expected_var / trials)
    assert abs(outputs.mean() - increment) < 5 * stderr
    assert abs(outputs.var() / expected_var - 1) < .025
    print(f"DARE: empirical mean={outputs.mean():.5f}, variance={outputs.var():.5f}; exact mean=.20000, variance={expected_var:.5f}")

    # The row number is an index; token meaning establishes correspondence.
    vocab_a, vocab_b = ["red", "blue", "<sql>"], ["blue", "red", "<code>"]
    ea = np.array([[1., 0.], [0., 1.], [2., 1.]])
    eb = np.array([[0., 1.], [1., 0.], [-1., 2.]])
    naive = (ea + eb) / 2
    assert not np.allclose(naive[0], [1., 0.])
    union = sorted(set(vocab_a + vocab_b))
    aligned_rows = []
    for token in union:
        rows = [table[vocab.index(token)] for vocab, table in [(vocab_a, ea), (vocab_b, eb)] if token in vocab]
        aligned_rows.append(np.mean(rows, axis=0))
    output = dict(zip(union, aligned_rows))
    np.testing.assert_allclose(output["red"], [1., 0.])
    np.testing.assert_allclose(output["<sql>"], [2., 1.])
    print("Vocabulary: token-aligned red =", output["red"], "; naive row 0 =", naive[0])

    # Train five small regressors on independent noisy samples of the same
    # underlying task. The last run has a deliberately shifted training target.
    truth = np.array([1., -2., .5])
    models = []
    for i in range(5):
        xtrain = rng.normal(size=(24, 3))
        ytrain = xtrain @ (truth + (1.5 if i == 4 else 0)) + rng.normal(0, .8, 24)
        models.append(np.linalg.lstsq(xtrain, ytrain, rcond=None)[0])
    xval = rng.normal(size=(128, 3))
    yval = xval @ truth + rng.normal(0, .8, 128)
    xtest = rng.normal(size=(4096, 3))
    ytest = xtest @ truth + rng.normal(0, .8, 4096)
    soup, selected = greedy_soup(models, xval, yval)
    best = min(models, key=lambda w: mse(xval, yval, w))
    assert mse(xval, yval, soup) <= mse(xval, yval, best) + 1e-12
    np.testing.assert_allclose(soup, np.mean([models[i] for i in selected], axis=0))
    print("Greedy selected runs:", selected)
    print(f"Validation MSE: best={mse(xval,yval,best):.5f}; soup={mse(xval,yval,soup):.5f}")
    print(f"Independent test MSE: best={mse(xtest,ytest,best):.5f}; soup={mse(xtest,ytest,soup):.5f}; uniform={mse(xtest,ytest,np.mean(models,axis=0)):.5f}")
    print("PASS: permutation, TIES, base/delta algebra, DARE moments, token alignment, held-out selection")


if __name__ == "__main__":
    main()
