#!/usr/bin/env python3
"""RoPE correctness lab. Python 3 + NumPy; no weights or GPU required.

Run: python3 lab.py --tokens 13 --dim 8 --chunk 3 --scale 4
Change --chunk or --scale; cached/full equality must survive. The two
intentional bugs must still be detected. These are numerical checks, not
long-context quality measurements or GPU benchmarks.
"""
import argparse
import math
import numpy as np


def frequencies(dim, base=10000.0):
    assert dim > 0 and dim % 2 == 0
    return base ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)


def rope(x, positions, inv_freq):
    """x [tokens, heads, dim], adjacent pairs; float64 reference arithmetic."""
    x = np.asarray(x, dtype=np.float64)
    phase = np.asarray(positions, dtype=np.float64)[:, None, None] * inv_freq
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * np.cos(phase) - b * np.sin(phase)
    out[..., 1::2] = a * np.sin(phase) + b * np.cos(phase)
    return out


def matrix_reference(x, positions, inv_freq):
    """Independent scalar 2x2 rotation, not a call to vectorized rope()."""
    out = np.empty_like(x)
    for t, pos in enumerate(positions):
        for h in range(x.shape[1]):
            for j, freq in enumerate(inv_freq):
                angle = pos * freq
                rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                     [math.sin(angle), math.cos(angle)]])
                out[t, h, 2*j:2*j+2] = rotation @ x[t, h, 2*j:2*j+2]
    return out


def attention(q, k, v, query_positions, key_positions):
    scores = np.einsum('thd,shd->hts', q, k) / math.sqrt(q.shape[-1])
    allowed = np.asarray(key_positions)[None, :] <= np.asarray(query_positions)[:, None]
    scores = np.where(allowed[None, :, :], scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum('hts,shd->thd', weights, v)


def chunked(q, k, v, inv_freq, chunk_size, wrong_positions=False):
    keys, values, outputs = [], [], []
    for start in range(0, len(q), chunk_size):
        end = min(start + chunk_size, len(q))
        # Rotation positions and mask positions are logically separate.
        pos = np.arange(start, end)
        rotation_pos = np.arange(end-start) if wrong_positions else pos
        qr = rope(q[start:end], rotation_pos, inv_freq)
        keys.append(rope(k[start:end], rotation_pos, inv_freq))
        values.append(v[start:end])
        outputs.append(attention(qr, np.concatenate(keys), np.concatenate(values),
                                 pos, np.arange(end)))
    return np.concatenate(outputs)


def yarn_paper_frequencies(inv_freq, original_length, scale, low=1., high=32.):
    """Paper's ramp in rotations r; library implementations may ramp by index."""
    rotations = original_length * inv_freq / (2 * np.pi)
    gamma = np.clip((rotations-low)/(high-low), 0., 1.)
    return inv_freq * (gamma + (1-gamma)/scale)


def run(args):
    rng = np.random.default_rng(19)
    q, k, v = [rng.normal(size=(args.tokens, 2, args.dim)) for _ in range(3)]
    pos = np.arange(args.tokens)
    freq = frequencies(args.dim) / args.scale
    rotated = rope(q, pos, freq)
    matrix_error = np.max(np.abs(rotated-matrix_reference(q, pos, freq)))
    norm_error = np.max(np.abs((rotated**2).sum(-1)-(q**2).sum(-1)))
    assert matrix_error < 1e-12 and norm_error < 1e-12
    full = attention(rope(q,pos,freq), rope(k,pos,freq), v, pos, pos)
    cached = chunked(q,k,v,freq,args.chunk)
    cache_error = np.max(np.abs(full-cached))
    shifted = attention(rope(q,pos+37,freq), rope(k,pos+37,freq),v,pos,pos)
    shift_error = np.max(np.abs(full-shifted))
    assert cache_error < 1e-12 and shift_error < 1e-12
    bad = chunked(q,k,v,freq,args.chunk,wrong_positions=True)
    bad_error = np.max(np.abs(full-bad))
    assert bad_error > 1e-6, 'Choose chunk < tokens to expose the injected reset bug.'
    split = args.tokens // 2
    old_freq, new_freq = frequencies(args.dim), frequencies(args.dim)/4
    stale = np.concatenate([rope(k[:split],pos[:split],old_freq),
                            rope(k[split:],pos[split:],new_freq)])
    correct_keys = rope(k,pos,new_freq)
    q_last = rope(q[-1:],pos[-1:],new_freq)
    stale_output = attention(q_last,stale,v,pos[-1:],pos)
    ref_output = attention(q_last,correct_keys,v,pos[-1:],pos)
    stale_error = np.max(np.abs(stale_output-ref_output))
    assert stale_error > 1e-6
    # This repair is for fixed projected q/k/v at one attention layer only.
    # It does not recompute earlier hidden states in a multilayer decoder.
    repaired = attention(q_last,rope(k,pos,new_freq),v,pos[-1:],pos)
    assert np.allclose(repaired,ref_output,atol=1e-12,rtol=0)
    assert np.float16(2048) == np.float16(2049)
    y = yarn_paper_frequencies(frequencies(args.dim),4096,args.scale)
    assert np.all(y <= frequencies(args.dim)+1e-15)
    assert np.all(y >= frequencies(args.dim)/args.scale-1e-15)
    print('PASS: 2x2 reference, norms, relative positions, chunked KV parity')
    print(f'matrix_error={matrix_error:.2e}; cache_error={cache_error:.2e}; shift_error={shift_error:.2e}')
    print(f'DETECTED: reset_position_error={bad_error:.6f}; stale_frequency_error={stale_error:.6f}')
    print('PASS: float16 position collision detected; YaRN paper ramp bounds')
    print('Scope: one attention layer with fixed projected Q/K/V; no quality benchmark.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens',type=int,default=13)
    parser.add_argument('--dim',type=int,default=8)
    parser.add_argument('--chunk',type=int,default=3)
    parser.add_argument('--scale',type=float,default=4.)
    args = parser.parse_args()
    if args.tokens < 4 or args.dim < 2 or args.dim % 2 or not 1 <= args.chunk < args.tokens or not 1 <= args.scale <= 16:
        parser.error('Require tokens >= 4, positive even dim, 1 <= chunk < tokens, 1 <= scale <= 16.')
    run(args)
