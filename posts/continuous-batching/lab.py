#!/usr/bin/env python3
"""确定性 CPU 调度教学模拟；首个 token 在最后一段 prefill 后产出。"""
import argparse
from dataclasses import dataclass, field


@dataclass
class Request:
    name: str
    arrival: float
    prompt: int
    output: int
    remaining: int = field(init=False)
    emitted: list = field(default_factory=list)

    def __post_init__(self):
        self.remaining = self.prompt


def simulate(budget, verbose=False):
    requests = [Request("A", 0, 6, 3), Request("B", 2, 2, 2), Request("C", 6, 3, 2)]
    now, trace = 0.0, []
    while any(len(r.emitted) < r.output for r in requests):
        ready = [r for r in requests if r.arrival <= now and len(r.emitted) < r.output]
        if not ready:
            now = min(r.arrival for r in requests if r.arrival > now)
            continue
        # 每轮开始确定批次；轮内到达的请求在下一轮再考虑。
        decode = sorted([r for r in ready if r.remaining == 0], key=lambda r: (r.emitted[-1], r.arrival))[:budget]
        remaining_budget = budget - len(decode)
        prefill = []
        for r in ready:
            if r.remaining and remaining_budget:
                chunk = min(r.remaining, remaining_budget)
                prefill.append((r, chunk))
                remaining_budget -= chunk
        p, d = sum(n for _, n in prefill), len(decode)
        assert 0 < p + d <= budget
        finish = now + 2 + 0.5 * p + d  # 单位 ms；教学成本模型，不是 GPU 基准。
        for r in decode:
            r.emitted.append(finish)
        for r, n in prefill:
            r.remaining -= n
            if r.remaining == 0:
                r.emitted.append(finish)
        trace.append((now, finish, p, d))
        if verbose:
            print(f"{now:g}->{finish:g} ms: P={p} D={d} prefill="
                  + str([(r.name, n) for r, n in prefill]) + " decode=" + str([r.name for r in decode]))
        now = finish
    assert all(r.remaining == 0 and len(r.emitted) == r.output for r in requests)
    assert sum(p for _, _, p, _ in trace) == 11
    assert sum(d for _, _, _, d in trace) == 4
    return requests, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=4)
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("budget 必须为正")
    # 独立手算时间轴，不根据模拟结果构造期望值。
    checked, trace = simulate(4)
    assert [r.emitted for r in checked] == [[8, 13, 16.5], [8, 13], [16.5, 19.5]]
    assert [e for _, e, _, _ in trace] == [4, 8, 13, 16.5, 19.5]
    requests, trace = simulate(args.budget, verbose=True)
    for r in requests:
        itl = [b - a for a, b in zip(r.emitted, r.emitted[1:])]
        print(f"{r.name}: tokens={r.emitted} TTFT={r.emitted[0] - r.arrival:g} ms ITL={itl}")
    output_count = sum(r.output for r in requests)
    print(f"output_throughput={1000 * output_count / trace[-1][1]:.2f} token/s")
    print("PASS: exact hand-worked timeline, token conservation, budget, completion")


if __name__ == "__main__":
    main()
