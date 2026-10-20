# 最小推理引擎 · 实验记录

课程：https://hs-black.github.io/posts/mini-inference-engine/
学习路径：https://hs-black.github.io/posts/llm-engineering/

## 运行环境

- 日期 / Python / NumPy：
- CPU / 线程设置：
- 代码版本或文件哈希：
- 命令：
- 权重：固定种子随机初始化；token 是数值 ID，不评价语言能力。

## 1. 正确性基线

运行 `python3 lab.py`。记录 dense / cached logits 的最大绝对误差。
解释为什么要检查 logits，而不只检查 argmax token。

| 页大小 | batch | 最大误差 | 生成 ID 是否一致 | 结束后空闲页 |
| --- | --- | --- | --- | --- |
| 1 | 1 | | | |
| 2 | 2 | | | |
| 4 | 4 | | | |

增加一个跨越两页边界、长度不同的新请求，重新验证。
独立基线 `dense()` 不经过分页地址转换。

## 2. 容量与生命周期

每 token 的 KV 字节 = 2 × 层数 × KV 头数 × 头维度 × 元素字节。
默认 NumPy float64：2 × 2 × 2 × 8 × 8 = 512 B/token。
每页 4 token：2048 B/page。
池中 3 页：6144 B，另有模型、Python 元数据和临时 gather 张量。

- 我的页大小与池大小：
- 预测最大可承诺 token 槽位：
- `prompt + max_new - 1` 为什么减一：
- 预留额度、实际已分配页面、物理数组容量分别是多少：
- 为什么还有空页时，FIFO 队头也可能暂时不能准入：

## 3. 调度时间线

运行 `python3 lab.py --demo`。
tick 是逻辑迭代编号，没有 GPU 毫秒含义。

| 请求 | 到达 tick | 第一输出 tick | TTFT（tick） | 输出间隔（tick） | 完成 tick |
| --- | --- | --- | --- | --- | --- |
| A | | | | | |
| B | | | | | |
| C | | | | | |
| D | | | | | |

只改变一个变量：batch / blocks / page_size / 到达时间。
预测哪个请求受益，运行后解释差异。
本实现 prompt 也逐 token 推进；并行 chunk prefill 是后续扩展。

## 4. 三个故障

依次运行：

```sh
python3 lab.py --fault position
python3 lab.py --fault mask
python3 lab.py --fault leak
```

| 故障 | 最小复现输入 | 最先失败的检查 | 违反了哪个不变量 | 修复位置 |
| --- | --- | --- | --- | --- |
| 位置全部清零 | | | | |
| 去掉 padding mask | | | | |
| 请求结束不释放 KV | | | | |

提示：mask 故障需要同 batch 内历史长度不同。
请增加自己的故障：例如 KV head 映射错误，或错误复用其他请求的物理页。

## 5. 本机 CPU 测量

运行 `python3 lab.py --benchmark`。
记录七次运行的中位数与最小值；预热不计入计时。
这是 float64 NumPy 小模型的端到端 CPU 用时，不能外推 GPU 吞吐。

- 受控变量：
- 正确性检查：
- 运行前预测：
- 测量结果：
- 是否支持预测：
- 下一次实验：

## 6. 进阶扩展（任选一项）

- Chunk prefill：一个请求一次处理多个 token，显式构造全局 causal mask。
- Prefix sharing：页引用计数、写时复制、模型/adapter/位置配置的完整缓存身份。
- 结构化采样：维护可回滚 grammar 状态，验证每个中间前缀合法。
- 调度策略：在相同请求与容量下比较 FIFO 和长度感知准入，记录公平性。
- GPU 后端：先用 PyTorch 对齐 logits，再用 Triton 替换一个算子。

验收：保留独立参考实现；覆盖长度不齐、页边界、失败路径；先正确，再解释测量。
