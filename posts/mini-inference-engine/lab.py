#!/usr/bin/env python3
"""A runnable teaching inference engine. Python 3.10+, NumPy >=1.24.

python3 lab.py                    # independent dense/cache/scheduler checks
python3 lab.py --demo             # scheduling trace, numerical token IDs
python3 lab.py --benchmark        # actual local CPU wall time, not GPU throughput
python3 lab.py --fault position   # demonstrate detection of a position bug
python3 lab.py --fault mask       # demonstrate detection of a padding-mask bug
python3 lab.py --fault leak       # demonstrate detection of a page leak

Two-layer randomly initialized decoder: RMSNorm, GQA, RoPE, SwiGLU.
Float64 makes correctness comparisons easy. No tokenizer or trained weights.
The scheduler processes one input token per selected request per iteration,
including during prefill. It batches the matrix multiplications, then gathers
paged KV into padded CPU arrays; it is not a fused paged GPU attention kernel.
Arrival timestamps below are logical ticks, never claimed as milliseconds.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import json
import math
import time
import numpy as np


@dataclass(frozen=True)
class Config:
    layers: int = 2
    width: int = 32
    heads: int = 4
    kv_heads: int = 2
    head_dim: int = 8
    hidden: int = 64
    vocab: int = 64


def rms(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def softmax(x):
    z = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def rope(x, positions):
    """x [..., heads, d]; positions [...] with adjacent real/imaginary pairs."""
    d = x.shape[-1]
    angles = np.asarray(positions)[..., None, None] * (
        10000.0 ** (-np.arange(0, d, 2) / d)
    )
    a, b = x[..., 0::2], x[..., 1::2]
    y = np.empty_like(x)
    y[..., 0::2] = a * np.cos(angles) - b * np.sin(angles)
    y[..., 1::2] = a * np.sin(angles) + b * np.cos(angles)
    return y


class TinyModel:
    def __init__(self, seed=7, cfg=Config()):
        self.cfg = cfg
        assert cfg.width == cfg.heads * cfg.head_dim
        assert cfg.heads % cfg.kv_heads == 0 and cfg.head_dim % 2 == 0
        rng = np.random.default_rng(seed)
        def weight(m, n):
            return rng.normal(size=(m, n)) / math.sqrt(m)
        self.embed = weight(cfg.vocab, cfg.width)
        self.output = weight(cfg.width, cfg.vocab)
        self.blocks = [{
            "q": weight(cfg.width, cfg.width),
            "k": weight(cfg.width, cfg.kv_heads * cfg.head_dim),
            "v": weight(cfg.width, cfg.kv_heads * cfg.head_dim),
            "o": weight(cfg.width, cfg.width),
            "gate": weight(cfg.width, cfg.hidden),
            "up": weight(cfg.width, cfg.hidden),
            "down": weight(cfg.hidden, cfg.width),
        } for _ in range(cfg.layers)]

    @staticmethod
    def mlp(x, w):
        z = rms(x)
        gate = z @ w["gate"]
        return x + (gate / (1 + np.exp(-gate)) * (z @ w["up"])) @ w["down"]

    def dense(self, tokens):
        """Independent full-sequence path: lower-triangular T×T attention."""
        c = self.cfg
        x = self.embed[np.asarray(tokens)]
        t = len(tokens)
        assert t > 0
        allowed = np.arange(t)[:, None] >= np.arange(t)[None, :]
        for w in self.blocks:
            z = rms(x)
            q = rope((z @ w["q"]).reshape(t, c.heads, c.head_dim), np.arange(t))
            k = rope((z @ w["k"]).reshape(t, c.kv_heads, c.head_dim), np.arange(t))
            v = (z @ w["v"]).reshape(t, c.kv_heads, c.head_dim)
            k = np.repeat(k, c.heads // c.kv_heads, axis=1)
            v = np.repeat(v, c.heads // c.kv_heads, axis=1)
            score = np.einsum("thd,shd->hts", q, k) / math.sqrt(c.head_dim)
            prob = softmax(np.where(allowed[None], score, -np.inf))
            y = np.einsum("hts,shd->thd", prob, v).reshape(t, c.width)
            x = self.mlp(x + y @ w["o"], w)
        return rms(x) @ self.output

    def step_batch(self, tokens, caches, fault=None):
        """One new position per request, shared matrix operations across B."""
        c = self.cfg
        b = len(tokens)
        if b != len(caches) or b == 0:
            raise ValueError("one input token per cache is required")
        if len({id(cache) for cache in caches}) != b:
            raise ValueError("a cache may appear only once in a batch")
        if any(not isinstance(token, (int, np.integer)) or
               token < 0 or token >= c.vocab for token in tokens):
            raise ValueError("input token must be an integer inside the vocabulary")
        # Check the entire allocation before mutating any cache or page pool.
        # This CPU executor is synchronous; concurrent executors need a lock.
        needed = {}
        for cache in caches:
            if len(cache.table) < math.ceil((cache.length + 1) / cache.pool.page_size):
                needed[cache.pool] = needed.get(cache.pool, 0) + 1
        if any(count > len(pool.free) for pool, count in needed.items()):
            raise MemoryError("batch cannot reserve all required KV pages")
        positions = np.array([cache.length for cache in caches])
        for cache in caches:
            cache.reserve_next()  # allocate once, before writing all layers
        x = self.embed[np.asarray(tokens)]
        for layer, w in enumerate(self.blocks):
            z = rms(x)
            pos = np.zeros(b, dtype=int) if fault == "position" else positions
            q = rope((z @ w["q"]).reshape(b, c.heads, c.head_dim), pos)
            k = rope((z @ w["k"]).reshape(b, c.kv_heads, c.head_dim), pos)
            v = (z @ w["v"]).reshape(b, c.kv_heads, c.head_dim)
            length = int(positions.max()) + 1
            keys = np.zeros((b, length, c.kv_heads, c.head_dim))
            values = np.zeros_like(keys)
            for i, cache in enumerate(caches):
                cache.write(layer, k[i], v[i])
                keys[i, :positions[i]+1], values[i, :positions[i]+1] = cache.read_with_current(layer)
            keys = np.repeat(keys, c.heads // c.kv_heads, axis=2)
            values = np.repeat(values, c.heads // c.kv_heads, axis=2)
            score = np.einsum("bhd,bthd->bht", q, keys) / math.sqrt(c.head_dim)
            mask = np.arange(length)[None, :] <= positions[:, None]
            if fault != "mask":
                score = np.where(mask[:, None], score, -np.inf)
            prob = softmax(score)
            y = np.einsum("bht,bthd->bhd", prob, values).reshape(b, c.width)
            x = self.mlp(x + y @ w["o"], w)
        for cache in caches:
            cache.length += 1
        return rms(x) @ self.output


class PagePool:
    def __init__(self, cfg, blocks=16, page_size=4):
        if blocks < 1 or page_size < 1:
            raise ValueError("blocks and page_size must be positive")
        self.page_size = page_size
        # Each physical block contains all layers for page_size token positions.
        self.data = np.zeros((blocks, cfg.layers, 2, page_size, cfg.kv_heads, cfg.head_dim))
        self.free = list(reversed(range(blocks)))
        self.owners = {}
        self.peak = 0

    def allocate(self, owner):
        if not self.free:
            raise MemoryError("KV page pool exhausted")
        block = self.free.pop()
        assert block not in self.owners
        self.owners[block] = owner
        self.peak = max(self.peak, len(self.owners))
        return block

    def release(self, block, owner):
        assert self.owners[block] == owner
        del self.owners[block]
        self.data[block] = 0  # hygiene, correctness must still rely on valid lengths
        self.free.append(block)


class PagedCache:
    def __init__(self, pool, owner):
        self.pool, self.owner = pool, owner
        self.table = []
        self.length = 0

    def reserve_next(self):
        if len(self.table) < math.ceil((self.length + 1) / self.pool.page_size):
            self.table.append(self.pool.allocate(self.owner))

    def write(self, layer, k, v):
        page, slot = divmod(self.length, self.pool.page_size)
        block = self.table[page]
        assert self.pool.owners[block] == self.owner
        self.pool.data[block, layer, 0, slot] = k
        self.pool.data[block, layer, 1, slot] = v

    def read_with_current(self, layer):
        """Read history plus the just-written token, before length is committed."""
        # Gather a logical prefix from non-contiguous physical blocks.
        n = self.length + 1
        chunks = [self.pool.data[block, layer] for block in self.table]
        joined = np.concatenate(chunks, axis=1)
        return joined[0, :n], joined[1, :n]

    def release(self):
        for block in self.table:
            self.pool.release(block, self.owner)
        self.table.clear()
        self.length = 0


def sample(logits, rng, temperature=0.0, top_k=0, top_p=1.0):
    if (not np.isfinite(temperature) or temperature < 0 or
        not np.isfinite(top_p) or not (0 < top_p <= 1) or
        not isinstance(top_k, (int, np.integer)) or top_k < 0):
        raise ValueError("temperature>=0, top_k>=0, 0<top_p<=1 required")
    logits = np.asarray(logits, dtype=np.float64)
    if (logits.ndim != 1 or not logits.size or np.any(np.isnan(logits)) or
        np.any(np.isposinf(logits)) or not np.any(np.isfinite(logits))):
        raise ValueError("one-dimensional logits with at least one finite value required")
    if temperature == 0:
        return int(np.argmax(logits))
    order = np.argsort(-logits, kind="stable")
    if top_k:
        order = order[:top_k]
    # Shift before scaling so large positive logits / small temperature cannot
    # overflow to +inf. A very negative gap may safely saturate to -inf.
    with np.errstate(over="ignore", under="ignore"):
        if temperature < 1:
            scaled = (logits[order] - logits[order[0]]) / temperature
        else:
            # At large T, scale first: opposite near-max floats may otherwise
            # overflow during subtraction even though their scaled gap is small.
            scaled = logits[order] / temperature
            scaled -= scaled[0]
        prob = softmax(scaled)
    # Include the first token that reaches the cumulative threshold.
    keep = min(len(order), int(np.searchsorted(np.cumsum(prob), top_p, side="left")) + 1)
    prob = prob[:keep]
    prob /= prob.sum()
    return int(rng.choice(order[:keep], p=prob))


@dataclass
class Request:
    name: str
    prompt: list[int]
    max_new: int
    arrival: int = 0
    generated: list[int] = field(default_factory=list)
    emitted: list[int] = field(default_factory=list)
    cache: PagedCache | None = None
    credit: int = 0


def run_engine(model, requests, blocks=16, page_size=4, batch=2,
               temperature=0.0, fault=None):
    """Round-robin batching with conservative worst-case admission credits.

    Credits reserve capacity, while pages are physically allocated on demand.
    Prefill is deliberately tokenwise to keep one common batched code path.
    """
    if batch < 1:
        raise ValueError("batch must be positive")
    pool = PagePool(model.cfg, blocks, page_size)
    if len({r.name for r in requests}) != len(requests):
        raise ValueError("request names must be unique")
    for r in requests:
        if not r.prompt or r.max_new < 1 or r.arrival < 0:
            raise ValueError("non-empty prompt, max_new>=1, arrival>=0 required")
        if any(t < 0 or t >= model.cfg.vocab for t in r.prompt):
            raise ValueError("prompt token outside vocabulary")
        if r.cache is not None or r.generated:
            raise ValueError("requests are single-use; create fresh Request objects")
        r.credit = math.ceil((len(r.prompt) + r.max_new - 1) / page_size)
        if r.credit > blocks:
            raise ValueError(f"{r.name} needs {r.credit} pages but pool has {blocks}")
    # Independent RNG per request avoids scheduling order changing random draws.
    rngs = {r.name: np.random.default_rng(100+i) for i, r in enumerate(requests)}
    pending = sorted(requests, key=lambda r: r.arrival)
    ready, active = deque(), deque()
    tick, reserved, trace = 0, 0, []
    while pending or ready or active:
        while pending and pending[0].arrival <= tick:
            ready.append(pending.pop(0))
        # FIFO admission; deliberately conservative and may cause head-of-line waits.
        while ready and reserved + ready[0].credit <= blocks:
            r = ready.popleft()
            r.cache = PagedCache(pool, r.name)
            reserved += r.credit
            active.append(r)
        if not active:
            assert not ready, "credits should be released when requests finish"
            tick = max(tick, pending[0].arrival)
            continue
        selected = [active.popleft() for _ in range(min(batch, len(active)))]
        inputs = [r.prompt[r.cache.length] if r.cache.length < len(r.prompt)
                  else r.generated[-1] for r in selected]
        logits = model.step_batch(inputs, [r.cache for r in selected], fault)
        row = {"tick": tick, "batch": [r.name for r in selected],
               "pages": len(pool.owners), "reserved": reserved, "emitted": []}
        for r, z in zip(selected, logits):
            if r.cache.length >= len(r.prompt):
                r.generated.append(sample(z, rngs[r.name], temperature))
                r.emitted.append(tick + 1)  # end of logical iteration
                row["emitted"].append(r.name)
            if len(r.generated) == r.max_new:
                if fault != "leak":
                    r.cache.release()
                reserved -= r.credit
            else:
                active.append(r)
        trace.append(row)
        assert len(pool.owners) <= reserved, "live KV pages exceed active reservation credits"
        tick += 1
    assert not pool.owners and len(pool.free) == blocks, "KV pages leaked"
    return trace, pool


def dense_generate(model, prompt, count):
    history = list(prompt)
    out = []
    for _ in range(count):
        token = int(np.argmax(model.dense(history)[-1]))
        out.append(token)
        history.append(token)
    return out


def fixture():
    return [
        Request("A", [1, 2, 3], 4, 0),
        Request("B", [8], 3, 0),
        Request("C", [3, 5, 7, 9, 11], 2, 2),
        Request("D", [4, 6], 3, 3),
    ]


def verify(fault=None):
    model = TinyModel()
    # Unequal lengths force padding, several block boundaries, reallocated pages.
    pool = PagePool(model.cfg, blocks=12, page_size=2)
    a, b = PagedCache(pool, "A"), PagedCache(pool, "B")
    histories = [[], []]
    largest = 0.0
    for i in range(9):
        selected = [(a, 0, (3*i+1) % model.cfg.vocab)]
        if i % 2 == 0:
            selected.append((b, 1, (2*i+2) % model.cfg.vocab))
        rows = model.step_batch([s[2] for s in selected],
                                [s[0] for s in selected], fault)
        for (_, idx, token), logits in zip(selected, rows):
            histories[idx].append(token)
            reference = model.dense(histories[idx])[-1]
            largest = max(largest, float(np.max(np.abs(reference - logits))))
            np.testing.assert_allclose(logits, reference, atol=1e-10, rtol=1e-10,
                                       err_msg=f"dense/cache mismatch at step {i}, request {idx}")
    a.release()
    b.release()
    assert len(pool.free) == 12
    print(f"PASS dense vs paged batched logits: max error {largest:.2e}")
    # A future input must never change earlier dense logits.
    before = model.dense([1, 3, 5, 7])
    after = model.dense([1, 3, 5, 31])
    np.testing.assert_allclose(before[:3], after[:3], atol=1e-12)
    print("PASS causal mask: future token cannot change earlier logits")
    for size in (1, 2, 4, 7):
        for batch in (1, 2, 4):
            requests = fixture()
            _, pool = run_engine(model, requests, blocks=16, page_size=size,
                                 batch=batch, fault=fault)
            for r in requests:
                assert r.generated == dense_generate(model, r.prompt, r.max_new)
            assert len(pool.free) == 16
    print("PASS 12 page-size/batch combinations: identical greedy outputs, no leaks")
    # Force queuing under the worst-case admission budget.
    requests = fixture()
    trace, pool = run_engine(model, requests, blocks=3, page_size=4, batch=2, fault=fault)
    assert pool.peak <= 3
    for r in requests:
        assert r.generated == dense_generate(model, r.prompt, r.max_new)
    print(f"PASS bounded admission: peak {pool.peak}/3 pages, {len(trace)} logical iterations")
    # Allocation exhaustion is handled before corrupting any page owner.
    tiny = PagePool(model.cfg, blocks=1, page_size=1)
    cache = PagedCache(tiny, "X")
    model.step_batch([1], [cache])
    try:
        cache.reserve_next()
    except MemoryError:
        assert tiny.owners == {0: "X"}
    else:
        raise AssertionError("expected allocation failure")
    cache.release()
    a, b = PagedCache(tiny, "A"), PagedCache(tiny, "B")
    try:
        model.step_batch([1, 2], [a, b])
    except MemoryError:
        assert not a.table and not b.table and not tiny.owners
    else:
        raise AssertionError("expected transactional batch allocation failure")
    try:
        model.step_batch([1, 2], [a, a])
    except ValueError:
        assert a.length == 0 and not tiny.owners
    else:
        raise AssertionError("duplicate cache accepted")
    model.step_batch([1], [a])  # a failed batch must remain retryable
    a.release()
    print("PASS OOM boundary: atomic reservation, duplicate rejection, clean retry")
    # Sampling checks an analytic distribution, independent of model internals.
    rng = np.random.default_rng(42)
    logits = np.log([0.6, 0.3, 0.1])
    draws = [sample(logits, rng, 1.0, top_p=0.8) for _ in range(12000)]
    freq = np.bincount(draws, minlength=3) / len(draws)
    np.testing.assert_allclose(freq, [2/3, 1/3, 0], atol=0.02, rtol=0)
    assert sample([1e308, -1e308], rng, 1e-5) == 0
    edge_rng, reference_rng = np.random.default_rng(42), np.random.default_rng(42)
    for _ in range(20):
        assert sample([1e308, -1e308], edge_rng, 1e308) == reference_rng.choice(
            2, p=softmax(np.array([1.0, -1.0])))
    try:
        sample([1, 2], rng, float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("NaN temperature accepted")
    print("PASS top-p sampling: frequencies match [2/3, 1/3, 0]")


def demo():
    model, requests = TinyModel(), fixture()
    trace, pool = run_engine(model, requests, blocks=3, batch=2)
    print("Logical ticks only; input token IDs and outputs have no language meaning.")
    for row in trace:
        print(json.dumps(row))
    for r in requests:
        print(f"{r.name}: output={r.generated}, first_token={r.emitted[0]-r.arrival} ticks, "
              f"gaps={np.diff(r.emitted).tolist()} ticks")
    print(f"pool_bytes={pool.data.nbytes}; peak_allocated_pages={pool.peak}")


def benchmark():
    model = TinyModel()
    # Warm-up is excluded. End-to-end local CPU includes scheduling and allocation.
    run_engine(model, fixture())
    samples = []
    for _ in range(7):
        start = time.perf_counter()
        run_engine(model, fixture())
        samples.append(time.perf_counter() - start)
    print("LOCAL CPU WALL TIME, float64 NumPy, random tiny model, 12 output tokens/run.")
    print(f"median_seconds={np.median(samples):.6f}; min_seconds={min(samples):.6f}")
    print("Do not use this CPU toy throughput to predict GPU or production capacity.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--fault", choices=["position", "mask", "leak"])
    args = parser.parse_args()
    if args.fault:
        try:
            verify(args.fault)
        except (AssertionError, MemoryError) as exc:
            message = next((line for line in str(exc).splitlines() if line.strip()), "invariant failed")
            print(f"EXPECTED DETECTION [{args.fault}]: {message}")
        else:
            raise AssertionError("fault injection escaped the checks")
    elif args.demo:
        demo()
    elif args.benchmark:
        benchmark()
    else:
        verify()


if __name__ == "__main__":
    main()
