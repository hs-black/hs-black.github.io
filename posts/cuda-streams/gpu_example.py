#!/usr/bin/env python3
"""Real CUDA example: three streams, two reusable device-buffer slots.
Requires a CUDA-enabled PyTorch installation and NVIDIA GPU.
Run: python3 gpu_example.py --chunks 8 --elements 1048576
No speedup is assumed: inspect a profiler trace to verify actual overlap.
"""
import argparse
import time
import torch


def run(chunks, elements):
    h_in = [torch.arange(elements, dtype=torch.float32, pin_memory=True).remainder_(97)
            for _ in range(chunks)]
    h_out = [torch.empty(elements, dtype=torch.float32, pin_memory=True) for _ in range(chunks)]
    d_in = [torch.empty(elements, device='cuda') for _ in range(2)]
    d_out = [torch.empty(elements, device='cuda') for _ in range(2)]
    upload, compute, download = [torch.cuda.Stream() for _ in range(3)]
    uploaded = [torch.cuda.Event() for _ in range(chunks)]
    computed = [torch.cuda.Event() for _ in range(chunks)]
    downloaded = [torch.cuda.Event() for _ in range(chunks)]
    # Even empty allocations may reuse storage touched on the current stream.
    for stream in (upload, compute, download):
        stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(chunks):
            slot = i % 2
            with torch.cuda.stream(upload):
                if i >= 2:
                    upload.wait_event(downloaded[i-2])
                d_in[slot].copy_(h_in[i], non_blocking=True)
                uploaded[i].record(upload)
            with torch.cuda.stream(compute):
                compute.wait_event(uploaded[i])
                torch.mul(d_in[slot], 2, out=d_out[slot])
                d_out[slot].add_(1)
                computed[i].record(compute)
            with torch.cuda.stream(download):
                download.wait_event(computed[i])
                h_out[i].copy_(d_out[slot], non_blocking=True)
                downloaded[i].record(download)
        # The final D2H event also follows every earlier D2H on this stream.
        downloaded[-1].synchronize()
    wall_ms = (time.perf_counter() - start) * 1000
    # All host and device tensor references remain alive until completion.
    for source, result in zip(h_in, h_out):
        torch.testing.assert_close(result, source*2+1, rtol=0, atol=0)
    return wall_ms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--chunks', type=int, default=8)
    p.add_argument('--elements', type=int, default=1048576)
    a = p.parse_args()
    if min(a.chunks, a.elements) <= 0:
        p.error('chunks/elements must be positive')
    if not torch.cuda.is_available():
        raise SystemExit('A CUDA GPU and CUDA-enabled PyTorch are required.')
    print('PyTorch:', torch.__version__, 'GPU:', torch.cuda.get_device_name())
    run(2, min(a.elements, 4096))   # context/kernel warmup; not a performance baseline
    wall = run(a.chunks, a.elements)
    print(f'PASS: all output values equal 2*x+1; enqueue-through-completion wall time={wall:.3f} ms')
    print('This includes CPU submission and transfers; compare an equivalent serial baseline separately.')


if __name__ == '__main__':
    main()
