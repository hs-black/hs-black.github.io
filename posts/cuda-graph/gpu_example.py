#!/usr/bin/env python3
"""真实 CUDA Graph 最小实验。需要 PyTorch（CUDA 版本）与 NVIDIA GPU。

python3 gpu_example.py
验证多轮 copy/replay、固定输入指针、输出覆盖与 clone。此示例演示语义，
不用于证明任何性能收益；小张量计时会被提交和测量开销主导。
"""
import torch


def main():
    if not torch.cuda.is_available():
        raise SystemExit('This example requires CUDA-enabled PyTorch and an NVIDIA GPU.')
    torch.manual_seed(0)
    static_x = torch.zeros(4, device='cuda')

    def f(x):
        return x.square() + 1

    with torch.inference_mode():
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                f(static_x)
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_y = f(static_x)
        input_ptr, output_ptr = static_x.data_ptr(), static_y.data_ptr()

        static_x.fill_(3)
        graph.replay()
        old_alias = static_y
        saved = static_y.clone()
        static_x.fill_(4)
        graph.replay()
        # assert_close / .cpu() 会等待结果；它们在捕获外，用于验证。
        torch.testing.assert_close(old_alias, torch.full_like(static_y, 17))
        torch.testing.assert_close(saved, torch.full_like(static_y, 10))
        for _ in range(5):
            fresh_x = torch.randn_like(static_x)
            expected = f(fresh_x)
            static_x.copy_(fresh_x)
            graph.replay()
            torch.testing.assert_close(static_y, expected)
            assert static_x.data_ptr() == input_ptr
            assert static_y.data_ptr() == output_ptr
        torch.cuda.synchronize()
    print('CUDA graph correctness: PASS (5 varying inputs, stable pointers, output alias/clone)')
    print('This is a correctness experiment, not a GPU performance benchmark.')


if __name__ == '__main__':
    main()
