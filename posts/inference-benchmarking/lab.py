#!/usr/bin/env python3
"""CPU 排队实验；全部时间来自教学输入，不是 GPU benchmark。

python3 lab.py
python3 lab.py --rate 12
python3 lab.py --arrival closed --stall-ms 500
python3 lab.py --arrival open --stall-ms 500 --json

一个 FCFS worker，一次处理一个请求。每个请求输出 4 个 token；
首 token 用 40% 服务时间，后 3 个间隔各用 20%。以此隔离排队效果。
实际 LLM 会 batching，必须用实测服务曲线替换此模型后才能估计容量。
仅 Python 标准库；不会联网、下载模型或访问 GPU。
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
import json
import math
import random
from statistics import mean

@dataclass(frozen=True)
class Record:
    arrival: float
    start: float
    tokens: tuple[float, ...]
    success: bool = True

    @property
    def finish(self):
        return self.tokens[-1]

    @property
    def ttft(self):
        return self.tokens[0] - self.arrival

    @property
    def gaps(self):
        return [b-a for a, b in zip(self.tokens, self.tokens[1:])]

    @property
    def tpot(self):
        return mean(self.gaps) if self.gaps else None


def percentile(values, q):
    """Nearest-rank percentile: sorted[ceil(q*N)-1]，q 在 (0,1]。"""
    if not values or not 0 < q <= 1:
        raise ValueError("需要非空样本和 0 < q <= 1")
    return sorted(values)[math.ceil(q*len(values))-1]


def simulate(rate=8., count=60, service=.1, arrival="open", pattern="regular", stall=0.):
    rng = random.Random(7)
    records = []
    next_arrival = 0.
    free = 0.
    for i in range(count):
        if i:
            if arrival == "closed":
                next_arrival = records[-1].finish  # 单客户端，上一请求结束才提交下一请求。
            else:
                next_arrival += 1/rate if pattern == "regular" else rng.expovariate(rate)
        start = max(next_arrival, free)
        extra = stall if i == 0 else 0.
        tokens = tuple(start + extra + f*service for f in (.4, .6, .8, 1.))
        records.append(Record(next_arrival, start, tokens))
        free = tokens[-1]
    return records


def summarize(records, ttft_slo=.2, tpot_slo=.03):
    # 时间窗从首个计划到达到最后一个请求完成，包含 drain；失败不进入 goodput。
    # 单 token 的 TPOT 不适用：该项不做门槛检查，分位数只统计有定义的样本。
    if not records:
        raise ValueError("至少需要一个请求记录")
    duration = max(r.finish for r in records) - min(r.arrival for r in records)
    if duration <= 0:
        raise ValueError("观察窗口必须为正")
    def quantile_or_none(values):
        return percentile(values, .99) if values else None
    successful = [r for r in records if r.success]
    good = [r for r in successful if r.ttft <= ttft_slo+1e-12
            and (r.tpot is None or r.tpot <= tpot_slo+1e-12)]
    itls = [gap for r in successful for gap in r.gaps]
    return {
        "offered": len(records), "completed": len(successful), "good": len(good),
        "duration_s": duration,
        "request_rps": len(successful)/duration,
        "output_tps": sum(len(r.tokens) for r in successful)/duration,
        "goodput_rps": len(good)/duration,
        "p99_ttft_ms": quantile_or_none([r.ttft*1000 for r in successful]),
        "p99_tpot_ms": quantile_or_none([r.tpot*1000 for r in successful if r.tpot is not None]),
        "p99_itl_ms": quantile_or_none([gap*1000 for gap in itls]),
        "mean_in_flight": sum(r.finish-r.arrival for r in records)/duration,
    }


def checks():
    # 可手算的独立时间戳，而非用模拟器生成 oracle。
    a = Record(0., .02, (.10, .12, .20, .22))
    assert math.isclose(a.ttft, .10)
    assert all(math.isclose(x, y) for x, y in zip(a.gaps, [.02, .08, .02]))
    assert math.isclose(a.tpot, .04)
    assert percentile([1, 2, 3, 4, 100], .8) == 4
    # 规则到达的 FCFS 递推有闭式：start_i=max(i/rate, i*service)。
    for rate in (8., 12.):
        rs = simulate(rate)
        for i, r in enumerate(rs):
            assert math.isclose(r.start, max(i/rate, i*.1), abs_tol=1e-10)
        for left, right in zip(rs, rs[1:]):
            assert right.start >= left.finish-1e-12
        assert all(r.start >= r.arrival-1e-12 for r in rs)
    low, high = summarize(simulate(8)), summarize(simulate(12))
    assert low["good"] == 60 and high["good"] == 10
    assert math.isclose(high["duration_s"], 6.)
    assert high["output_tps"] > low["output_tps"]
    assert high["goodput_rps"] < low["goodput_rps"]
    # 失败项仍占观察窗口，不能被 goodput 的分母抹掉。
    failed = Record(0., 0., (.04, .06, .08, .1), False)
    good = Record(.1, .1, (.14, .16, .18, .2))
    summary = summarize([failed, good])
    assert summary["completed"] == 1 and summary["good"] == 1
    assert math.isclose(summary["goodput_rps"], 5.)
    all_failed = summarize([failed])
    assert all_failed["completed"] == all_failed["good"] == 0
    assert all_failed["request_rps"] == all_failed["goodput_rps"] == 0
    assert all_failed["p99_ttft_ms"] is None
    assert all_failed["p99_tpot_ms"] is None and all_failed["p99_itl_ms"] is None
    single = summarize([Record(0., 0., (.1,))])
    assert single["completed"] == single["good"] == 1
    assert math.isclose(single["p99_ttft_ms"], 100.)
    assert single["p99_tpot_ms"] is None and single["p99_itl_ms"] is None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rate", type=float, default=8.)
    p.add_argument("--requests", type=int, default=60)
    p.add_argument("--service-ms", type=float, default=100.)
    p.add_argument("--arrival", choices=["open", "closed"], default="open")
    p.add_argument("--pattern", choices=["regular", "poisson"], default="regular")
    p.add_argument("--stall-ms", type=float, default=0.)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    if a.rate <= 0 or a.requests < 1 or a.service_ms <= 0 or a.stall_ms < 0:
        p.error("rate/service-ms/requests 必须为正；stall-ms 不得为负")
    checks()
    records = simulate(a.rate, a.requests, a.service_ms/1000, a.arrival, a.pattern, a.stall_ms/1000)
    s = summarize(records)
    if a.json:
        print(json.dumps({"teaching_simulation": True, "summary": s,
                          "records": [asdict(r) for r in records]}, indent=2))
    else:
        print("checks: PASS (timestamps, FCFS oracle, goodput, failures)")
        print(f"teaching simulation: {a.arrival}/{a.pattern}; {a.requests} requests")
        print(f"completed={s['completed']} good={s['good']} duration={s['duration_s']:.3f} s")
        print(f"throughput={s['request_rps']:.3f} req/s output={s['output_tps']:.3f} tok/s")
        print(f"goodput={s['goodput_rps']:.3f} req/s p99_TTFT={s['p99_ttft_ms']:.3f} ms")
        print(f"p99_TPOT={s['p99_tpot_ms']:.3f} ms p99_ITL={s['p99_itl_ms']:.3f} ms")
        print(f"time-average in-flight={s['mean_in_flight']:.3f}")

if __name__ == "__main__":
    main()
