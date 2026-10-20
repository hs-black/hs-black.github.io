#!/usr/bin/env python3
"""Image/video token accounting, patch-order checks and cache identity.

Run: python3 -m pip install numpy
     python3 lab.py
     python3 lab.py --frames 8 --height 448 --width 448

CPU teaching experiment. Input dimensions are ALREADY resized to a multiple
of patch_size * merge_size. This is not a full Qwen image processor or model.
The patch example averages values to expose layout; a learned vision encoder
and patch merger perform different computations in a real model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class VisionShape:
    frames: int
    height: int
    width: int
    patch: int = 14
    merge: int = 2
    temporal: int = 2

    def grids(self) -> tuple[int, int, int]:
        if min(self.frames, self.height, self.width, self.patch, self.merge, self.temporal) < 1:
            raise ValueError("shape and patch sizes must be positive")
        if self.height % (self.patch * self.merge) or self.width % (self.patch * self.merge):
            raise ValueError("pass already-resized H/W divisible by patch * merge")
        return ((self.frames + self.temporal - 1) // self.temporal,
                self.height // self.patch, self.width // self.patch)

    def tokens(self) -> tuple[int, int]:
        t, h, w = self.grids()
        return t * h * w, t * h * w // self.merge**2


def patchify(x: np.ndarray, patch: int, temporal: int) -> np.ndarray:
    """[F,H,W,C] -> [Tg,Hg,Wg,temporal,patch,patch,C]."""
    f, h, w, c = x.shape
    if h % patch or w % patch:
        raise ValueError("H/W must be divisible by patch")
    padding = (-f) % temporal
    if padding:
        x = np.concatenate([x, np.repeat(x[-1:], padding, axis=0)])
    tg = x.shape[0] // temporal
    return x.reshape(tg, temporal, h // patch, patch, w // patch, patch, c).transpose(0, 2, 4, 1, 3, 5, 6)


def merge_mean(patches: np.ndarray, merge: int) -> np.ndarray:
    """Teaching-only merger: one mean per spatial merge x merge group."""
    features = patches.mean(axis=(3, 4, 5, 6))
    t, h, w = features.shape
    if h % merge or w % merge:
        raise ValueError("patch grid must be divisible by merge")
    return features.reshape(t, h // merge, merge, w // merge, merge).mean(axis=(2, 4))


def kv_bytes(tokens: int, layers: int = 28, kv_heads: int = 4,
             head_dim: int = 128, element_bytes: int = 2) -> int:
    """Dense retained K and V only; excludes allocator, weights, activations."""
    return tokens * 2 * layers * kv_heads * head_dim * element_bytes


def cache_key(media: bytes, *, processor: dict, vision_revision: str, vision_adapter_revision: str = "none",
              timestamps: tuple[float, ...] = ()) -> str:
    # Bytes + exact preprocessing/model context, not just URL or image filename.
    context = json.dumps({"processor": processor, "vision_revision": vision_revision,
                          "vision_adapter_revision": vision_adapter_revision, "timestamps": timestamps}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(len(context).to_bytes(8, "big") + context + media).hexdigest()


def prefix_key(embedding_key: str, text_ids: list[int], *, lm_revision: str,
               adapter_revision: str, position_ids: list[int], rope_config: str = "default",
               visibility: str = "causal") -> str:
    payload = [embedding_key, text_ids, lm_revision, adapter_revision, position_ids, rope_config, visibility]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def self_test() -> None:
    # Independent slicing verifies ordering, including temporal end padding.
    x = np.arange(3 * 8 * 8, dtype=np.float64).reshape(3, 8, 8, 1)
    padded = np.concatenate([x, x[-1:]], axis=0)
    patches = patchify(x, 2, 2)
    for t in range(2):
        for y in range(4):
            for z in range(4):
                np.testing.assert_array_equal(patches[t, y, z], padded[t*2:(t+1)*2, y*2:(y+1)*2, z*2:(z+1)*2])
    merged = merge_mean(patches, 2)
    expected = np.array([[[padded[t*2:(t+1)*2, y*4:(y+1)*4, z*4:(z+1)*4].mean()
                           for z in range(2)] for y in range(2)] for t in range(2)])
    np.testing.assert_allclose(merged, expected)
    assert VisionShape(1, 448, 672).tokens() == (1536, 384)
    assert VisionShape(8, 448, 448).tokens() == (4096, 1024)
    assert kv_bytes(384) == 22_020_096  # 21 MiB, independent numeric answer
    assert kv_bytes(1) == 57_344
    proc = {"height": 448, "width": 672, "patch": 14, "merge": 2}
    key = cache_key(b"same image bytes", processor=proc, vision_revision="vision-v1")
    assert key == cache_key(b"same image bytes", processor=dict(reversed(list(proc.items()))), vision_revision="vision-v1")
    assert key != cache_key(b"same image bytes", processor={**proc, "width": 896}, vision_revision="vision-v1")
    assert key != cache_key(b"same image bytes", processor=proc, vision_revision="vision-v2")
    assert key != cache_key(b"same image bytes", processor=proc, vision_revision="vision-v1", vision_adapter_revision="vision-adapter-v1")
    assert key != cache_key(b"same image bytes", processor=proc, vision_revision="vision-v1", timestamps=(0.0, 1.0))
    prefix = dict(lm_revision="lm-v1", adapter_revision="adapter-a", position_ids=[0, 1, 2])
    pkey = prefix_key(key, [7, 8, 9], **prefix)
    assert pkey != prefix_key(key, [7, 8, 10], **prefix)
    assert pkey != prefix_key(key, [7, 8, 9], **{**prefix, "adapter_revision": "adapter-b"})
    assert pkey != prefix_key(key, [7, 8, 9], **prefix, rope_config="linear-2")
    assert pkey != prefix_key(key, [7, 8, 9], **prefix, visibility="window-2")
    print("CHECK patch ordering / temporal padding / merge: PASS")
    print("CHECK token budget / KV bytes / cache identity: PASS")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=int, default=1)
    p.add_argument("--height", type=int, default=448)
    p.add_argument("--width", type=int, default=672)
    p.add_argument("--text", type=int, default=256, help="text tokens excluding four teaching-example delimiters")
    p.add_argument("--output", type=int, default=128)
    args = p.parse_args()
    if args.text < 0 or args.output < 0:
        p.error("text/output must be nonnegative")
    self_test()
    shape = VisionShape(args.frames, args.height, args.width)
    raw, visual = shape.tokens()
    prompt = args.text + 4 + visual
    reserved = prompt + args.output
    print(f"grid_t,h,w={shape.grids()} raw_patches={raw} visual_tokens={visual}")
    print(f"prompt_tokens={prompt} reserved_tokens={reserved}")
    print(f"visual_KV={kv_bytes(visual) / 2**20:.5f} MiB")
    print(f"reserved_KV={kv_bytes(reserved) / 2**20:.5f} MiB")
    dense_slots = prompt * prompt
    useful_pairs = prompt * (prompt + 1) // 2
    print(f"causal_score_slots={dense_slots} useful_pairs={useful_pairs}")
    print("Model: Qwen2.5-VL-7B text KV dimensions; explicit 4 delimiters are a toy input convention.")
    print("Try doubled H/W, then doubled frames; compare visual tokens before total request cost.")


if __name__ == "__main__":
    main()
