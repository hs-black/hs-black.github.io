#!/usr/bin/env python3
"""Toy text curation checks: exact MinHash probability, LSH, semantic chain.
Run: python3 lab.py --seed 7 --trials 30000
Dependencies: NumPy. Every random stream is seeded; no network or models.
"""
import argparse
import hashlib
import math
import numpy as np


def shingles(text, n=3):
    words = text.lower().split()
    return {tuple(words[i:i+n]) for i in range(len(words)-n+1)}


def exact_jaccard(a, b):
    return len(a & b) / len(a | b)


def stable_id(text):
    return hashlib.sha256(text.encode()).hexdigest()


def semantic_keep(vectors, threshold, retained_only=False):
    kept = []
    for j, z in enumerate(vectors):
        prior = kept if retained_only else range(j)
        if all(float(vectors[i] @ z) <= threshold for i in prior):
            kept.append(j)
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--trials', type=int, default=30000)
    ap.add_argument('--semantic-threshold', type=float, default=.9)
    args = ap.parse_args()
    if args.trials < 1000 or not 0 <= args.semantic_threshold <= 1:
        ap.error('trials must be >= 1000; threshold must lie in [0,1]')
    rng = np.random.default_rng(args.seed)
    original = 'a small model reads useful text and learns shared patterns'
    edited = 'a small model reads useful text and learns novel patterns'
    a, b = shingles(original), shingles(edited)
    assert stable_id(original) == stable_id(original)
    assert stable_id(original) != stable_id(edited)
    truth = exact_jaccard(a, b)
    # Each random permutation is truly uniform over the finite union.
    universe = sorted(a | b)
    ai = [i for i, v in enumerate(universe) if v in a]
    bi = [i for i, v in enumerate(universe) if v in b]
    collisions = 0
    for _ in range(args.trials):
        ranks = rng.permutation(len(universe))
        collisions += ranks[ai].min() == ranks[bi].min()
    empirical = collisions / args.trials
    tolerance = 6 * math.sqrt(truth*(1-truth)/args.trials) + 1/args.trials
    assert abs(empirical-truth) < tolerance
    print(f'Text Jaccard={truth:.6f}; uniform-permutation collision={empirical:.6f}')
    # Independent Bernoulli matches isolate the banding probability from hashing.
    matches = rng.random((args.trials, 14, 8)) < .75
    observed = matches.all(axis=2).any(axis=1).mean()
    predicted = 1-(1-.75**8)**14
    tol = 6 * math.sqrt(predicted*(1-predicted)/args.trials) + 1/args.trials
    assert abs(observed-predicted) < tol
    print(f'LSH b=14,r=8,s=.75: analytic={predicted:.6f}; simulated={observed:.6f}')
    angles = np.deg2rad([0,20,40])
    vectors = np.column_stack([np.cos(angles),np.sin(angles)])
    all_prior = semantic_keep(vectors, args.semantic_threshold)
    retained = semantic_keep(vectors, args.semantic_threshold, True)
    # Independent geometric closed form checks the default counterexample.
    assert np.allclose(vectors @ vectors.T, np.cos(angles[:,None]-angles[None,:]))
    if math.cos(math.radians(40)) <= args.semantic_threshold < math.cos(math.radians(20)):
        assert all_prior == [0] and retained == [0,2]
    if args.semantic_threshold >= math.cos(math.radians(20)):
        assert all_prior == retained == [0,1,2]
    names = lambda ids: [chr(65+i) for i in ids]
    print('Semantic all-prior:', names(all_prior), '; retained-only:', names(retained))
    p, retention = np.array([.6,.3,.1]), np.array([.2,.8,.9])
    mass = p*retention
    fractions = mass/mass.sum()
    assert np.allclose(fractions, [4/15,8/15,1/5])
    assert np.isclose(mass.sum(), .45)
    print('Retained domain shares:', np.round(fractions,6))
    gb = 100_000_000*14*8*8/10**9
    assert gb == 89.6
    print(f'100M documents, 112 uint64 hashes: {gb:.1f} GB (signature payload only)')
    print('All checks passed.')


if __name__ == '__main__':
    main()
