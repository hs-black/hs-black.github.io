#!/usr/bin/env python3
"""CUDA stream teaching model: dependency edges + exclusive resources.

Run: python3 lab.py [--chunks 4 --buffers 2 --h2d 2 --compute 5 --d2h 1]
All times are pedagogical milliseconds, NOT measurements of a GPU.
The scheduler uses fixed FIFO order on each resource, not an optimal scheduler.
"""
from dataclasses import dataclass
import argparse
import math


@dataclass(frozen=True)
class Job:
    name: str
    resource: str
    duration: float
    dependencies: tuple[str, ...]


def make_jobs(n, buffers, h2d, compute, d2h, *, serialized=False,
              upload_event=True, reuse_event=True):
    jobs = []
    for i in range(n):
        upload_deps = []
        if i >= buffers and reuse_event:
            upload_deps.append(f'D{i-buffers}')
        if serialized and i:
            upload_deps.append(f'D{i-1}')
        jobs.extend([
            Job(f'H{i}', 'H2D', h2d, tuple(upload_deps)),
            Job(f'C{i}', 'SM', compute, (f'H{i}',) if upload_event else ()),
            Job(f'D{i}', 'D2H', d2h, (f'C{i}',)),
        ])
    return jobs


def schedule(jobs):
    """Return {job: (start, end)}; append each job to its resource queue."""
    resource_free, times = {}, {}
    for job in jobs:
        assert job.duration > 0 and job.name not in times
        dependency_end = max((times[d][1] for d in job.dependencies), default=0)
        start = max(dependency_end, resource_free.get(job.resource, 0))
        times[job.name] = (start, start + job.duration)
        resource_free[job.resource] = start + job.duration
    return times


def hazards(times, n, buffers):
    """Check semantic data dependencies independently from scheduler edges."""
    result = []
    for i in range(n):
        if times[f'C{i}'][0] < times[f'H{i}'][1]:
            result.append(f'C{i} reads before H{i} completed')
        if times[f'D{i}'][0] < times[f'C{i}'][1]:
            result.append(f'D{i} reads before C{i} completed')
        if i >= buffers and times[f'H{i}'][0] < times[f'D{i-buffers}'][1]:
            result.append(f'H{i} overwrites slot {i % buffers} before D{i-buffers} completed')
    return result


def verify_resources(jobs, times):
    for resource in {j.resource for j in jobs}:
        spans = sorted(times[j.name] for j in jobs if j.resource == resource)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    for job in jobs:
        assert all(times[job.name][0] >= times[d][1] for d in job.dependencies)


def self_test():
    for n in (1, 2, 4, 9):
        # With an independent engine per stage and enough buffers, classic
        # pipeline makespan = sum(stage durations) + (n-1)*slowest stage.
        for h, c, d in ((2, 5, 1), (7, 2, 1), (1, 2, 8)):
            jobs = make_jobs(n, n, h, c, d)
            times = schedule(jobs)
            expected = h + c + d + (n-1) * max(h, c, d)
            assert math.isclose(max(t[1] for t in times.values()), expected)
            assert not hazards(times, n, n)
            verify_resources(jobs, times)
            one = schedule(make_jobs(n, 1, h, c, d))
            assert max(t[1] for t in one.values()) == n * (h+c+d)
    t = schedule(make_jobs(4, 2, 2, 5, 1))
    assert t == {'H0': (0, 2), 'C0': (2, 7), 'D0': (7, 8),
                 'H1': (2, 4), 'C1': (7, 12), 'D1': (12, 13),
                 'H2': (8, 10), 'C2': (12, 17), 'D2': (17, 18),
                 'H3': (13, 15), 'C3': (17, 22), 'D3': (22, 23)}
    bad_upload = schedule(make_jobs(4, 2, 2, 5, 1, upload_event=False))
    bad_reuse = schedule(make_jobs(4, 2, 2, 5, 1, reuse_event=False))
    assert any('reads before' in h for h in hazards(bad_upload, 4, 2))
    assert any('overwrites' in h for h in hazards(bad_reuse, 4, 2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--chunks', type=int, default=4)
    p.add_argument('--buffers', type=int, default=2)
    p.add_argument('--h2d', type=float, default=2)
    p.add_argument('--compute', type=float, default=5)
    p.add_argument('--d2h', type=float, default=1)
    a = p.parse_args()
    if min(a.chunks, a.buffers, a.h2d, a.compute, a.d2h) <= 0:
        p.error('all arguments must be positive')
    self_test()
    jobs = make_jobs(a.chunks, a.buffers, a.h2d, a.compute, a.d2h)
    times = schedule(jobs)
    verify_resources(jobs, times)
    assert not hazards(times, a.chunks, a.buffers)
    serial = a.chunks * (a.h2d + a.compute + a.d2h)
    pipeline = max(t[1] for t in times.values())
    print('Teaching simulation; durations are inputs, not GPU measurements.')
    print(f'serial={serial:.3f} ms pipeline={pipeline:.3f} ms speedup={serial/pipeline:.3f}x')
    for i in range(a.chunks):
        print(f'chunk {i} slot {i % a.buffers}: H={times[f"H{i}"]} C={times[f"C{i}"]} D={times[f"D{i}"]}')
    print('Missing upload event:', hazards(schedule(make_jobs(4, 2, 2, 5, 1, upload_event=False)), 4, 2)[0])
    print('Missing reuse event:', hazards(schedule(make_jobs(4, 2, 2, 5, 1, reuse_event=False)), 4, 2)[0])
    print('PASS: dependencies, resource exclusivity, closed-form bounds, and injected hazards')


if __name__ == '__main__':
    main()
