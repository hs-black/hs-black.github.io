#!/usr/bin/env python3
"""需要 PyTorch；CPU 即可。对比 dropout 模型开关 checkpoint 后的输出/梯度。

python3 torch_example.py
可把 preserve_rng_state=False 改在 checkpoint 调用中，观察梯度比较失败。
不要据此程序的 CPU 耗时推断 GPU 训练速度。
"""
import copy
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


torch.manual_seed(9)
base = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Dropout(.25), nn.Linear(16, 8)).double()
remat = copy.deepcopy(base)
data = torch.randn(4, 8, dtype=torch.float64)


def run(module, use_checkpoint):
    module.zero_grad(set_to_none=True)
    x = data.clone().requires_grad_(True)
    torch.manual_seed(42)  # 两次独立试验使用同一前向 dropout 掩码。
    if use_checkpoint:
        y = checkpoint(module, x, use_reentrant=False, preserve_rng_state=True)
    else:
        y = module(x)
    y.square().mean().backward()
    return y.detach(), x.grad.clone(), [p.grad.clone() for p in module.parameters()]


plain, checked = run(base, False), run(remat, True)
torch.testing.assert_close(plain[0], checked[0], rtol=1e-10, atol=1e-10)
torch.testing.assert_close(plain[1], checked[1], rtol=1e-10, atol=1e-10)
for a, b in zip(plain[2], checked[2]):
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
print('PASS: dropout output, input gradient and parameter gradients match')
