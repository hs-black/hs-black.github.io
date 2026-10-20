#!/usr/bin/env python3
"""CPU action contract and closed-loop timing lab; no robot dependencies.

Run: python3 -m pip install numpy
     python3 lab.py
     python3 lab.py --latency-ticks 5 --replan 4

A one-dimensional point follows x[k+1] = x[k] + dt*(u[k]+drift).
The planner predicts with zero drift. Computation latency is a chosen number
of simulated ticks, not a measured model runtime. On plan arrival, actions
whose timestamps already passed are discarded. If no valid action exists,
the simulated point receives a zero velocity command; drift remains. This is not a robot controller.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import numpy as np


def normalize(a: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    if np.any(high <= low):
        raise ValueError("each action range must have positive width")
    return np.clip(2 * (a - low) / (high - low) - 1, -1, 1)


def unnormalize(z: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return (z + 1) * (high - low) / 2 + low


def quantize(z: np.ndarray, bins: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Teaching quantizer: N true intervals, N+1 boundaries, N centers."""
    if bins < 2:
        raise ValueError("at least two bins")
    ids = np.minimum(((np.clip(z, -1, 1) + 1) * bins / 2).astype(int), bins - 1)
    return ids, -1 + (ids + 0.5) * 2 / bins


def frame_delta(delta: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]]) @ delta


def plan(x: float, target: float, horizon: int, dt: float, gain: float) -> np.ndarray:
    """A known linear predictor, deliberately not a learned VLA model."""
    out = []
    predicted = x
    for _ in range(horizon):
        u = gain * (target - predicted)
        out.append(u)
        predicted += dt * u
    return np.array(out)


@dataclass
class Result:
    rmse: float
    final_x: float
    missing_ticks: int
    requests: int
    max_inflight: int


def simulate(*, replan: int, latency: int, horizon: int = 20, steps: int = 200,
             dt: float = 0.02, gain: float = 2.0, drift: float = 0.10,
             switch_tick: int = 100) -> Result:
    if min(replan, horizon, steps) < 1 or latency < 0 or latency >= horizon:
        raise ValueError("require positive sizes and 0 <= latency < horizon")
    x = 0.0
    in_flight = []
    active: dict[int, float] = {}
    errors = []
    missing = requests = max_inflight = 0
    # Independent per-request compute slots are assumed here. max_inflight
    # exposes when replan frequency would overload a single serial worker.
    for tick in range(steps):
        target = 1.0 if tick < switch_tick else 0.5
        if tick % replan == 0:
            actions = plan(x, target, horizon, dt, gain)
            in_flight.append((tick + latency, tick, actions))
            requests += 1
        ready = [item for item in in_flight if item[0] <= tick]
        in_flight = [item for item in in_flight if item[0] > tick]
        # Completions at this timestamp release their compute slots before
        # counting concurrent outstanding work. Zero-latency jobs do not persist.
        max_inflight = max(max_inflight, len(in_flight))
        for _, captured_at, actions in ready:
            active = {captured_at + i: float(u) for i, u in enumerate(actions) if captured_at + i >= tick}
        u = active.pop(tick, None)
        if u is None:
            missing += 1
            u = 0.0
        x += dt * (u + drift)
        errors.append((x - target)**2)
    return Result(float(np.sqrt(np.mean(errors))), x, missing, requests, max_inflight)


def self_test() -> None:
    low, high = np.array([-.04]), np.array([.04])
    a = np.array([.013])
    z = normalize(a, low, high)
    ids, q = quantize(z)
    recovered = unnormalize(q, low, high)
    np.testing.assert_allclose(z, [.325], atol=1e-14)
    np.testing.assert_array_equal(ids, [5])
    np.testing.assert_allclose(recovered, [.015], atol=1e-14)
    assert abs(float(recovered[0] - a[0])) <= .005 + 1e-14
    edge_ids, _ = quantize(np.array([-1., 1.]))
    np.testing.assert_array_equal(edge_ids, [0, 7])
    np.testing.assert_allclose(frame_delta(np.array([.02, 0]), np.pi/2), [0, .02], atol=1e-14)
    # Closed-form independent reference for zero-delay feedback with drift.
    n, gain, dt, drift = 80, 2.0, .02, .10
    equilibrium = 1 + drift / gain
    expected = equilibrium * (1 - (1 - gain * dt)**n)
    actual = simulate(replan=1, latency=0, horizon=20, steps=n, gain=gain, dt=dt,
                      drift=drift, switch_tick=n+1)
    np.testing.assert_allclose(actual.final_x, expected, atol=1e-12)
    assert actual.missing_ticks == 0
    # All plans equal the exact no-disturbance trajectory, so chunk length
    # alone does not degrade it; disturbances/changes are what require feedback.
    a1 = simulate(replan=1, latency=0, drift=0, switch_tick=1000)
    a8 = simulate(replan=8, latency=0, drift=0, switch_tick=1000)
    np.testing.assert_allclose(a1.final_x, a8.final_x, atol=1e-12)
    slow = simulate(replan=4, latency=3)
    assert slow.missing_ticks == 3 and slow.max_inflight == 1
    assert simulate(replan=1, latency=3).max_inflight == 3
    print("CHECK normalization / quantization / frames: PASS")
    print("CHECK closed-form feedback / chunk equivalence / stale actions: PASS")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latency-ticks", type=int, default=3)
    p.add_argument("--replan", type=int, default=4)
    args = p.parse_args()
    self_test()
    for interval in sorted({args.replan, 16}):
        r = simulate(replan=interval, latency=args.latency_ticks)
        print(f"replan={interval} latency={args.latency_ticks} rmse={r.rmse:.6f} final_x={r.final_x:.6f} "
              f"missing_ticks={r.missing_ticks} requests={r.requests} max_inflight={r.max_inflight}")
    print("dt=20 ms; horizon=20; target changes at tick 100; drift=0.10 distance-unit/s.")
    print("A serial worker needs latency <= replan to sustain this request rate.")
    print("Try latency=0,3,5 and replan=1,4,16; inspect missing actions AND compute demand.")


if __name__ == "__main__":
    main()
