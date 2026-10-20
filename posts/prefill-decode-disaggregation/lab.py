#!/usr/bin/env python3
"""两张相同 GPU 的教学排队模型，全部耗时为给定输入，并非实测。

python3 lab.py
python3 lab.py --bandwidth-gib 0.125
python3 lab.py --prefill-ms 80 --decode-ms 80 --rate 12

比较：2 个 P+D 共置副本 vs 1 个 P worker + 1 个 D worker。
每个 worker 一次处理一个完整阶段；不模拟 batching 或 kernel overlap。
共享网络串行搬运，P / 网络 / D 可以处理不同请求而形成流水。
租约例子使用独立状态机，不把失败悄悄当成零成本传输。
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import math
from statistics import mean

@dataclass(frozen=True)
class Result:
    arrival: float
    p_start: float
    p_end: float
    x_start: float
    x_end: float
    d_start: float
    finish: float
    first: float


def kv_bytes(tokens, layers=32, kv_heads=8, head_dim=128, bytes_per_element=2):
    return 2*layers*tokens*kv_heads*head_dim*bytes_per_element


def colocated(arrivals, prefill, decode):
    available = [0., 0.]
    rows = []
    for arrival in arrivals:
        worker = min(range(2), key=lambda i: (available[i], i))
        start = max(arrival, available[worker])
        pe = start + prefill
        end = pe + decode
        available[worker] = end
        rows.append(Result(arrival, start, pe, pe, pe, pe, end, pe+decode/7))
    return rows


def disaggregated(arrivals, prefill, decode, transfer):
    p_free = x_free = d_free = 0.
    rows = []
    for arrival in arrivals:
        ps = max(arrival, p_free)
        pe = ps + prefill
        xs = max(pe, x_free)
        xe = xs + transfer
        ds = max(xe, d_free)
        end = ds + decode
        rows.append(Result(arrival, ps, pe, xs, xe, ds, end, ds+decode/7))
        p_free, x_free, d_free = pe, xe, end
    return rows


class Handoff:
    """保守协议：complete/ACK 必须早于 expires。源租约到期本身不破坏
    已验证的目标副本；这里把超期接受规则明确作为教学设计选择。
    真实系统还需要异步 I/O、完成记录与幂等 RPC。
    """
    def __init__(self, tokens, revision, expires):
        self.tokens = tuple(tokens)
        self.revision = revision
        self.expires = expires
        self.state = "READY"
        self.receiver_valid = False

    def begin(self):
        if self.state != "READY":
            raise ValueError("只能从 READY 开始复制")
        self.state = "COPYING"

    def complete(self, now, tokens, revision, all_blocks):
        if self.state != "COPYING":
            raise ValueError("没有正在进行的复制")
        valid = (now < self.expires and tuple(tokens) == self.tokens
                 and revision == self.revision and all_blocks)
        self.receiver_valid = valid
        self.state = "ACKED" if valid else "FAILED"
        return valid

    def release(self):
        if self.state not in {"ACKED", "FAILED"}:
            raise ValueError("复制未完成，不能释放源 block")
        self.state = "RELEASED"


def summarize(rows):
    duration = max(r.finish for r in rows) - min(r.arrival for r in rows)
    ttfts = [r.first-r.arrival for r in rows]
    return len(rows)/duration, mean(ttfts)*1000, max(ttfts)*1000


def checks():
    # 1 token、1 layer、1 head、2 维，K/V 各两个 BF16 => 8 bytes。
    assert kv_bytes(1, 1, 1, 2, 2) == 8
    assert kv_bytes(1024) == 128*1024**2
    # 2 个同时到达的请求：两个共置副本各自 200 ms 完成。
    c = colocated([0., 0.], .06, .14)
    assert all(math.isclose(r.finish, .20) for r in c)
    # 手排 P: [0,60],[60,120]; X: [60,80],[120,140]; D:[80,220],[220,360]。
    d = disaggregated([0., 0.], .06, .14, .02)
    assert math.isclose(d[0].finish, .22) and math.isclose(d[1].finish, .36)
    for i, r in enumerate(d):
        assert r.arrival <= r.p_start <= r.p_end <= r.x_start <= r.x_end <= r.d_start <= r.finish
        if i:
            assert r.p_start >= d[i-1].p_end-1e-12
            assert r.x_start >= d[i-1].x_end-1e-12
            assert r.d_start >= d[i-1].finish-1e-12
    for case, expected in [("ok", True), ("timeout", False), ("tokens", False),
                           ("revision", False), ("partial", False)]:
        h = Handoff([1, 2, 3], "model-a/rope-1", 1.)
        h.begin()
        ok = h.complete(2. if case == "timeout" else .5,
                        [1, 4, 3] if case == "tokens" else [1, 2, 3],
                        "model-b/rope-1" if case == "revision" else "model-a/rope-1",
                        case != "partial")
        assert ok == expected and h.receiver_valid == expected
        h.release()
        assert h.state == "RELEASED"
    h = Handoff([1], "v1", 1.)
    h.begin()
    try:
        h.release()
    except ValueError:
        pass
    else:
        raise AssertionError("源 block 被提前释放")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rate", type=float, default=8.)
    p.add_argument("--requests", type=int, default=20)
    p.add_argument("--prefill-ms", type=float, default=60.)
    p.add_argument("--decode-ms", type=float, default=140.)
    p.add_argument("--bandwidth-gib", type=float, default=8.)
    p.add_argument("--setup-ms", type=float, default=4.)
    p.add_argument("--prompt-tokens", type=int, default=1024)
    a = p.parse_args()
    if min(a.rate, a.requests, a.prefill_ms, a.decode_ms, a.bandwidth_gib, a.prompt_tokens) <= 0 or a.setup_ms < 0:
        p.error("输入必须为正；setup-ms 可以为零")
    checks()
    prefill, decode = a.prefill_ms/1000, a.decode_ms/1000
    size = kv_bytes(a.prompt_tokens)
    transfer = a.setup_ms/1000 + size/(a.bandwidth_gib*1024**3)
    arrivals = [i/a.rate for i in range(a.requests)]
    print("checks: PASS (KV bytes, schedule oracle, handoff identity/lease/partial-copy)")
    print(f"teaching simulation: same 2 GPUs; KV={size/1024**2:.3f} MiB; transfer={transfer*1000:.3f} ms")
    print(f"capacity bounds: colocated={2/(prefill+decode):.3f} req/s; PD={min(1/prefill,1/transfer,1/decode):.3f} req/s")
    for name, rows in [("colocated", colocated(arrivals, prefill, decode)),
                       ("PD", disaggregated(arrivals, prefill, decode, transfer))]:
        throughput, avg, maximum = summarize(rows)
        print(f"{name}: throughput={throughput:.3f} req/s mean_TTFT={avg:.3f} ms max_TTFT={maximum:.3f} ms")
    print("handoff: complete+matching+unexpired => valid; timeout/mismatch/partial => invalid")

if __name__ == "__main__":
    main()
