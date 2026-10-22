#!/usr/bin/env python3
"""Concept models: mathematical contracts on small CPU arrays.
Requires NumPy only. Run: python lab.py
These examples test distributions, pooling visibility and attention layouts;
they do not measure SONAR reconstruction or GPU performance.
"""
import numpy as np


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


def finite_gradient(function, x, eps=1e-6):
    answer = np.zeros_like(x)
    for i in range(len(x)):
        perturbation = np.zeros_like(x)
        perturbation[i] = eps
        answer[i] = (function(x + perturbation) - function(x - perturbation)) / (2 * eps)
    return answer


def mean_and_posterior():
    support, probability = np.array([-2., 2.]), np.array([.25, .75])
    mean = probability @ support
    variance = probability @ (support - mean)**2
    assert mean == 1 and variance == 3
    for estimate in np.linspace(-3, 3, 25):
        risk = probability @ (support - estimate)**2
        np.testing.assert_allclose(risk, variance + (estimate - mean)**2)
    # Bayes using Gaussian likelihoods independently checks the sigmoid formula.
    alpha, sigma = np.sqrt(.5), np.sqrt(.5)
    for observation in np.linspace(-2, 2, 21):
        log_likelihood = -(observation - alpha * support)**2 / (2 * sigma**2)
        posterior = softmax(np.log(probability) + log_likelihood)
        odds = np.log(.75 / .25) + 4 * alpha * observation / sigma**2
        closed = 1 / (1 + np.exp(-odds))
        np.testing.assert_allclose(posterior[1], closed, atol=1e-14)
        np.testing.assert_allclose(posterior @ support, 4 * closed - 2, atol=1e-14)
    print("MSE decomposition and Gaussian-likelihood posterior: passed")


def completed_concepts(h, probabilities, observed, threshold=.5):
    """A new boundary closes the preceding segment. EOS at T+1 closes last.
    h is a toy causal encoder output: each row depends only on that position.
    The test inspects which rows may contribute at a given observation time.
    """
    starts = np.flatnonzero(probabilities >= threshold)
    assert starts[0] == 0
    result = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(h)
        available = end + 1  # positions and observed use one-based numbering
        if available <= observed:
            result.append(h[start:end].mean(axis=0))
    return np.array(result)


def pooling_causality():
    values = np.arange(2., 18., 2.)[:, None]
    probabilities = np.array([1., .1, .2, .8, .1, .75, .1, .1])
    assert len(completed_concepts(values, probabilities, 3)) == 0
    np.testing.assert_allclose(completed_concepts(values, probabilities, 4), [[4.]])
    np.testing.assert_allclose(completed_concepts(values, probabilities, 6), [[4.], [9.]])
    np.testing.assert_allclose(completed_concepts(values, probabilities, 9), [[4.], [9.], [14.]])
    rng = np.random.default_rng(91)
    for observed in range(1, 9):
        changed = values.copy()
        changed[observed:] += rng.normal(0, 100, size=changed[observed:].shape)
        changed_probabilities = probabilities.copy()
        changed_probabilities[observed:] = rng.uniform(size=len(values) - observed)
        # Future values AND future boundary decisions may change freely.
        np.testing.assert_allclose(completed_concepts(values, probabilities, observed),
                                   completed_concepts(changed, changed_probabilities, observed))
    # A deliberately unsafe reference exposes how full-segment pooling can leak.
    altered = values.copy()
    altered[2] += 100  # future relative to observed=2
    assert values[:3].mean() != altered[:3].mean()
    print("closed-segment availability and future-perturbation invariance: passed")


def expanded_attention(scores, values, ids, allowed, compensate):
    visible_ids = ids[allowed]
    counts = np.bincount(visible_ids, minlength=len(scores))
    expanded_scores = scores[visible_ids].copy()
    if compensate:
        expanded_scores -= np.log(counts[visible_ids])
    return softmax(expanded_scores) @ values[visible_ids]


def multiplicity():
    scores, values = np.array([0., np.log(3.)]), np.array([2., 10.])
    unique = softmax(scores) @ values
    ids = np.array([0, 1, 1, 1])
    visible = np.ones(len(ids), dtype=bool)
    raw = expanded_attention(scores, values, ids, visible, False)
    corrected = expanded_attention(scores, values, ids, visible, True)
    np.testing.assert_allclose([unique, raw, corrected], [8., 9.2, 8.])
    rng = np.random.default_rng(7)
    for _ in range(40):
        count = rng.integers(1, 6, size=3)
        ids = np.repeat(np.arange(3), count)
        allowed = rng.uniform(size=len(ids)) > .35
        if not allowed.any():
            allowed[0] = True
        scores, values = rng.normal(size=3), rng.normal(size=3)
        present = np.unique(ids[allowed])
        reference = lambda logits: softmax(logits[present]) @ values[present]
        expanded = lambda logits: expanded_attention(logits, values, ids, allowed, True)
        np.testing.assert_allclose(expanded(scores), reference(scores), atol=1e-13)
        np.testing.assert_allclose(finite_gradient(expanded, scores),
                                   finite_gradient(reference, scores), atol=1e-9)
    # A truncated mask requires visible multiplicity, not the original segment size.
    ids = np.array([0, 0, 1, 1, 1, 1])
    allowed = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    scores, values = np.zeros(2), np.array([1., 7.])
    np.testing.assert_allclose(expanded_attention(scores, values, ids, allowed, True), 4.)
    visible_ids = ids[allowed]
    wrong = softmax(scores[visible_ids] - np.log(np.array([2., 4.])[visible_ids])) @ values[visible_ids]
    assert abs(wrong - 4) > .1
    print(f"KV copies: unique=8, repeated=9.2, corrected=8; truncated wrong-count result={wrong:.3f}")
    print("random masked layouts and score gradients: passed")


def auxiliary_and_margin():
    target, actual_rate, mean_probability = 4., .4, .35
    def auxiliary(g):
        return target / (target - 1) * ((target - 1) * actual_rate * g + (1 - actual_rate) * (1 - g)) - 1
    eps = 1e-6
    finite = (auxiliary(mean_probability + eps) - auxiliary(mean_probability - eps)) / (2 * eps)
    analytic = target / (target - 1) * (target * actual_rate - 1)
    np.testing.assert_allclose([finite, analytic], [.8, .8], atol=1e-9)
    # A small vector perturbation can cross a small-margin decoder decision.
    z, direction = np.array([1., .001]), np.array([0., 1.])
    changed = z - .002 * direction
    assert np.sign(direction @ z) != np.sign(direction @ changed)
    assert np.linalg.norm(changed - z) < .003
    print("compression-controller gradient=0.8; small-margin decoder boundary: passed")


def main():
    mean_and_posterior()
    pooling_causality()
    multiplicity()
    auxiliary_and_margin()
    print("all checks passed")


if __name__ == "__main__":
    main()
