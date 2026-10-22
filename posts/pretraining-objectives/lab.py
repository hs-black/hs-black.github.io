#!/usr/bin/env python3
"""CPU objective contracts: python lab.py. Requires NumPy; no model downloads."""
import numpy as np


def span_corrupt(tokens, spans):
    """Sorted, disjoint half-open spans. Sentinel strings are reserved tokens."""
    source, target, cursor = [], [], 0
    for i, (start, end) in enumerate(spans):
        assert cursor <= start < end <= len(tokens)
        marker = f"<s{i}>"
        source += tokens[cursor:start] + [marker]
        target += [marker] + tokens[start:end]
        cursor = end
    return source + tokens[cursor:], target + [f"<s{len(spans)}>"]


def reconstruct(source, target):
    spans, marker = {}, None
    for token in target:
        if token.startswith("<s"):
            marker = token
            spans[marker] = []
        else:
            spans[marker].append(token)
    return [token for item in source for token in spans.get(item, [item])]


def fim(tokens, left, right):
    assert 0 <= left < right <= len(tokens)
    return ["<PRE>"] + tokens[:left] + ["<SUF>"] + tokens[right:] + ["<MID>"] + tokens[left:right] + ["<EOS>"]


def recover_fim(stream):
    s, m, e = (stream.index(t) for t in ["<SUF>", "<MID>", "<EOS>"])
    return stream[1:s] + stream[m+1:e] + stream[s+1:m]


def target_pairs(doc_ids, horizon):
    """Rows are input anchors t; columns are prediction distances h=1..H.
    Packed documents occupy contiguous intervals; each new document has a new id.
    -1 marks an invalid target. EOS, if included, is part of its own document.
    """
    n = len(doc_ids)
    return np.array([[t+h if t+h < n and doc_ids[t] == doc_ids[t+h] else -1
                      for h in range(1, horizon+1)] for t in range(n)])


def loss_grad(features, heads, tokens, pairs):
    """Linear-head toy, not the paper's Transformer heads.
    features: [T,d]; heads: [H,d,V]; global valid-pair mean.
    Heads run sequentially and accumulate the shared-feature gradient.
    """
    count = np.sum(pairs >= 0)
    assert count > 0
    total, dz, dw = 0., np.zeros_like(features), np.zeros_like(heads)
    for h, weight in enumerate(heads):
        rows = np.flatnonzero(pairs[:, h] >= 0)
        labels = tokens[pairs[rows, h]]
        logits = features[rows] @ weight
        logits -= logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        total -= logp[np.arange(len(rows)), labels].sum() / count
        grad = np.exp(logp)
        grad[np.arange(len(rows)), labels] -= 1
        grad /= count
        dz[rows] += grad @ weight.T
        dw[h] = features[rows].T @ grad
    return total, dz, dw


def finite_difference(f, x, eps=1e-6):
    result = np.empty_like(x)
    for idx in np.ndindex(x.shape):
        before = x[idx]
        x[idx] = before + eps
        plus = f(x)
        x[idx] = before - eps
        minus = f(x)
        x[idx] = before
        result[idx] = (plus - minus) / (2 * eps)
    return result


def main():
    text = list("ABCDEFGH")
    for spans in [[(1, 3), (6, 7)], [(0, 1)], [(0, 8)]]:
        source, target = span_corrupt(text, spans)
        assert reconstruct(source, target) == text
        removed = sum(b-a for a, b in spans)
        assert len(source) == len(text)-removed+len(spans)
        assert len(target) == removed+len(spans)+1
    source, target = span_corrupt(text, [(1, 3), (6, 7)])
    print("T5 source:", " ".join(source), "| target:", " ".join(target))
    reordered = fim(list("ABCDEF"), 2, 4)
    assert recover_fim(reordered) == list("ABCDEF")
    chunks = [reordered[i:i+5] for i in range(0, len(reordered), 5)]
    assert not any(all(t in chunk for t in ["<PRE>", "<SUF>", "<MID>"]) for chunk in chunks)
    # Supervision is a shift in the transformed stream, independent of its original order.
    inputs, labels = reordered[:-1], reordered[1:]
    assert labels[inputs.index("<MID>")] == "C"
    docs = np.array([0, 0, 0, 0, 1, 1, 1])
    pairs = target_pairs(docs, 3)
    assert np.sum(pairs >= 0) == 9  # (3+2+1) + (2+1+0)
    assert np.all(pairs[3] == -1) and pairs[2, 1] == -1
    rng = np.random.default_rng(7)
    z = rng.normal(size=(7, 3)) * .2
    w = rng.normal(size=(3, 3, 5)) * .2
    tokens = np.array([0, 1, 2, 3, 1, 4, 2])
    loss, dz, dw = loss_grad(z, w, tokens, pairs)
    numeric_z = finite_difference(lambda x: loss_grad(x, w, tokens, pairs)[0], z)
    numeric_w = finite_difference(lambda x: loss_grad(z, x, tokens, pairs)[0], w)
    np.testing.assert_allclose(dz, numeric_z, atol=2e-9)
    np.testing.assert_allclose(dw, numeric_w, atol=2e-9)
    # Independently collect all valid (anchor, head, target) rows in one CE matrix.
    positions = np.argwhere(pairs >= 0)
    scores = np.stack([z[t] @ w[h] for t, h in positions])
    scores -= scores.max(axis=1, keepdims=True)
    logp = scores - np.log(np.exp(scores).sum(axis=1, keepdims=True))
    ref = -np.mean([logp[i, tokens[pairs[t, h]]] for i, (t, h) in enumerate(positions)])
    np.testing.assert_allclose(loss, ref)
    # Two microbatches: weight sums by their valid target counts.
    split = 4
    parts = [(z[:split], tokens[:split], target_pairs(docs[:split], 3)),
             (z[split:], tokens[split:], target_pairs(docs[split:], 3))]
    grads, counts = [], []
    for features, labels, targets in parts:
        grads.append(loss_grad(features, w, labels, targets)[2])
        counts.append(np.sum(targets >= 0))
    combined = sum(g*n for g, n in zip(grads, counts)) / sum(counts)
    np.testing.assert_allclose(combined, dw, atol=1e-14)
    assert np.max(np.abs(np.mean(grads, axis=0)-dw)) > 1e-4
    initial = loss
    for _ in range(120):
        loss, dz, dw = loss_grad(z, w, tokens, pairs)
        z -= .6 * dz
        w -= .6 * dw
    final = loss_grad(z, w, tokens, pairs)[0]
    assert final < initial * .3
    print(f"MTP valid pairs: {len(positions)}; feature/head gradients checked")
    print("Accumulation example: mean-of-means = 2.000; token mean = 1.400")
    print(f"Toy trainable features + linear heads: {initial:.4f} -> {final:.4f} nats/pair (120 steps)")
    print("PASS: reconstruction, shifted labels, packing counterexample, boundaries, gradients, accumulation")


if __name__ == "__main__":
    main()
