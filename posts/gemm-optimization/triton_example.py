#!/usr/bin/env python3
"""完整的教学 Triton GEMM，需 NVIDIA CUDA GPU（SM80+）与匹配的 PyTorch/Triton。

python3 triton_example.py

输入 FP16，FP32 accumulator，输出 FP16。支持正 stride 的二维 view。
比较独立 FP32 PyTorch 参考，并分别测量本 kernel 和 torch.mm 的中位数。
首次编译与分配不进入计时。这里没有预设性能结论。
"""
import torch
import triton
import triton.testing
import triton.language as tl


@triton.jit
def _matmul(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            SAM: tl.constexpr, SAK: tl.constexpr,
            SBK: tl.constexpr, SBN: tl.constexpr,
            SCM: tl.constexpr, SCN: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    mi = pm * BM + tl.arange(0, BM)
    nj = pn * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for block_k in range(tl.cdiv(K, BK)):
        ki = block_k * BK + kk
        a = tl.load(A + mi[:, None] * SAM + ki[None, :] * SAK,
                    mask=(mi[:, None] < M) & (ki[None, :] < K), other=0)
        b = tl.load(B + ki[:, None] * SBK + nj[None, :] * SBN,
                    mask=(ki[:, None] < K) & (nj[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
    out = acc.to(tl.float16)
    tl.store(C + mi[:, None] * SCM + nj[None, :] * SCN, out,
             mask=(mi[:, None] < M) & (nj[None, :] < N))


def launch(a, b, out):
    """out 须为无重叠的可写布局，且不得与 a/b 共享存储。

    下方调用者始终独立分配 out；正 stride 本身不能保证无重叠。
    """
    m, k = a.shape
    kb, n = b.shape
    assert k == kb and out.shape == (m, n)
    assert a.dtype == b.dtype == out.dtype == torch.float16
    assert a.is_cuda and b.is_cuda and out.is_cuda
    assert a.device == b.device == out.device
    assert all(x > 0 for x in a.stride() + b.stride() + out.stride())
    _matmul[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
        a, b, out, m, n, k, *a.stride(), *b.stride(), *out.stride(),
        BM=32, BN=32, BK=32, num_warps=4, num_stages=2)


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise SystemExit('需要 NVIDIA SM80+ CUDA GPU；CPU lab.py 可独立运行。')
    torch.manual_seed(7)
    # 禁用 FP32 reference 的 TF32，参考在 FP32 输入与累加下计算。
    torch.backends.cuda.matmul.allow_tf32 = False
    print('device:', torch.cuda.get_device_name())
    print('versions:', 'torch', torch.__version__, 'triton', triton.__version__)
    for m, n, k, mode in [(128, 256, 64, 'contiguous'),
                           (37, 53, 29, 'contiguous'),
                           (37, 53, 29, 'transposed'),
                           (37, 53, 29, 'offset-slice'),
                           (1, 127, 65, 'contiguous')]:
        if mode == 'transposed':
            a = torch.randn((k, m), device='cuda', dtype=torch.float16).t()
            b = torch.randn((n, k), device='cuda', dtype=torch.float16).t()
        elif mode == 'offset-slice':
            # 首元素偏移非零、行间有空隙；kernel 收到的指针已含 storage_offset。
            a = torch.randn((m + 2, k + 3), device='cuda', dtype=torch.float16)[1:1+m, 2:2+k]
            b = torch.randn((k + 2, n + 3), device='cuda', dtype=torch.float16)[1:1+k, 2:2+n]
            assert a.storage_offset() > 0 and b.storage_offset() > 0
        else:
            a = torch.randn((m, k), device='cuda', dtype=torch.float16)
            b = torch.randn((k, n), device='cuda', dtype=torch.float16)
        out = torch.empty((m, n), device='cuda', dtype=torch.float16)
        library = torch.empty_like(out)
        launch(a, b, out)
        reference = a.float() @ b.float()
        torch.testing.assert_close(out.float(), reference, atol=2e-2, rtol=2e-2)
        torch.mm(a, b, out=library)
        torch.testing.assert_close(out, library, atol=2e-2, rtol=2e-2)
        # do_bench 负责 warmup 与 CUDA event 同步。两侧都复用输出，排除分配成本。
        ours_ms = triton.testing.do_bench(lambda: launch(a, b, out), return_mode='median')
        library_ms = triton.testing.do_bench(lambda: torch.mm(a, b, out=library), return_mode='median')
        print(f'{m}x{n}x{k}, mode={mode}: PASS, '
              f'Triton={ours_ms:.4f} ms, torch.mm={library_ms:.4f} ms')
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
