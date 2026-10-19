#!/usr/bin/env python3
"""CPU 教学模拟：规范序列化的哈希链、满块边界、身份隔离与收益模型。"""
import argparse
import hashlib
import json


def chain(tokens, block_size=4, namespace="model-v1|adapter-0|tenant-A"):
    parent = hashlib.sha256(namespace.encode()).hexdigest()
    keys = []
    for start in range(0, len(tokens) - block_size + 1, block_size):
        payload = json.dumps([parent, tokens[start:start + block_size]], separators=(",", ":"))
        parent = hashlib.sha256(payload.encode()).hexdigest()
        keys.append(parent)
    return keys


def hit_tokens(tokens, cache, block_size=4, namespace="model-v1|adapter-0|tenant-A"):
    hits = 0
    for key in chain(tokens, block_size, namespace):
        if key not in cache:
            break
        hits += block_size
    # 只有 KV 而没有最终 logits 的接口，至少重算末 token 来产生 logits。
    return min(hits, (max(0, len(tokens) - 1) // block_size) * block_size)


def self_test():
    a = list(range(12))
    keys = chain(a)
    changed = a.copy()
    changed[5] = 99
    other = chain(changed)
    assert keys[0] == other[0] and keys[1] != other[1] and keys[2] != other[2]
    assert hit_tokens(a + [12], set(keys)) == 12
    assert hit_tokens(a, set(keys)) == 8  # 满 prompt 命中仍留出一个块重新算 logits。
    assert hit_tokens(changed, set(keys)) == 4
    assert hit_tokens(a, set(keys), namespace="model-v1|adapter-0|tenant-B") == 0
    assert hit_tokens(a, set(keys), namespace="model-v2|adapter-0|tenant-A") == 0
    assert hit_tokens(a, set(keys[1:])) == 0  # 后缀块存在，也不能跨过前面的缺口。
    assert len(chain(list(range(11)))) == 2
    # 理想 causal attention 的 query-key 对数：留存后缀必须继续读取历史。
    n, h = 4096, 3072
    all_pairs = n * (n + 1) // 2
    saved = h * (h + 1) // 2
    assert all_pairs == 8390656 and all_pairs - saved == 3670528


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--change-position", type=int, default=5)
    args = parser.parse_args()
    if args.block_size <= 0 or not 0 <= args.change_position < 13:
        parser.error("block-size > 0，change-position 在 0..12")
    self_test()
    original = list(range(13))
    changed = original.copy()
    changed[args.change_position] = 99
    cache = set(chain(original, args.block_size))
    hits = hit_tokens(changed, cache, args.block_size)
    expected = (args.change_position // args.block_size) * args.block_size
    assert hits == min(expected, (12 // args.block_size) * args.block_size)
    print(f"changed_position={args.change_position} reused={hits} recompute={13 - hits}")
    cold, hot = 55 + 4000 * 0.05, 55 + (4000 - 3072) * 0.05
    assert cold == 255 and abs(hot - 101.4) < 1e-12
    print(f"TTFT model: cold={cold:g} ms hot={hot:g} ms speedup={cold / hot:.3f}x")
    print("pair model: total=8390656 remaining=3670528 saved=56.25% (rounded)")
    print("PASS: causal hash chain, full-block boundary, identity isolation, missing-prefix stop, pair count")


if __name__ == "__main__":
    main()
