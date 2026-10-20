#!/usr/bin/env python3
"""Real torch.compile experiment (PyTorch with working Inductor toolchain).

TORCH_LOGS="graph_breaks,recompiles,guards" python3 torch_example.py --dynamic auto
python3 torch_example.py --dynamic false --device cuda
CPU works too; the first compilation may take substantial time.
"""
import argparse
import time
import torch


def function(x):
    y = (torch.sin(x) + 2.0).square()
    if x.shape[0] > 128:
        y = y / 2.0
    return y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dynamic', choices=('auto', 'true', 'false'), default='auto')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--iterations', type=int, default=20)
    a = p.parse_args()
    if a.iterations < 1:
        p.error('iterations must be positive')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA is not available.')
    dynamic = {'auto': None, 'true': True, 'false': False}[a.dynamic]
    torch.manual_seed(7)
    # compile() returns a wrapper; compilation normally starts on first input.
    compiled = torch.compile(function, dynamic=dynamic, fullgraph=True)
    sync = torch.cuda.synchronize if a.device == 'cuda' else lambda: None
    print('PyTorch:', torch.__version__, 'device:', a.device, 'dynamic:', dynamic)
    last = None
    with torch.inference_mode():
        for length in (64, 96, 128, 96, 64, 192, 129):
            x = torch.randn(length, 256, device=a.device)
            reference = function(x)
            sync()
            t0 = time.perf_counter()
            got = compiled(x)
            sync()
            elapsed = (time.perf_counter() - t0) * 1000
            torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-5)
            print(f'length={length} call_wall_ms={elapsed:.3f} correct')
            last = x
        # One already-seen input. This measures host call + work + final sync.
        sync()
        t0 = time.perf_counter()
        for _ in range(a.iterations):
            compiled(last)
        sync()
        warm_ms = (time.perf_counter() - t0) * 1000 / a.iterations
    print(f'warm_seen_length=129 mean_call_wall_ms={warm_ms:.3f}')
    print('First-call times include compilation; use TORCH_LOGS to count compilations and identify their causes.')


if __name__ == '__main__':
    main()
