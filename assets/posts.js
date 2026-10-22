// Every post on the site, with its category and tags. The blog page
// (/blog/) renders its timeline / category / series / tag views from this
// list, and post pages use it to show their tags and related posts.
//
// To add a post: add an entry at the TOP (newest first). kind is
// article | interactive | deck; cat is a key of CATEGORIES; reuse existing
// tags where possible. Then run `python3 scripts/post-links.py` to refresh
// the cross-reference index in assets/post-links.js.
(() => {
  const CATEGORIES = {
    training: '训练系统',
    inference: '推理系统',
    kernels: 'GPU 与算子',
    models: '模型架构与报告',
    recipes: '训练方法',
    basics: '入门与随笔',
  };
  const KINDS = { article: ['文章', 'kind'], interactive: ['交互', 'kind interactive'], deck: ['幻灯片', 'kind deck'] };

  const POSTS = [
    {"slug": "llm-engineering", "title": "LLM 工程学习路径：从张量到推理服务", "kind": "interactive", "cat": "basics", "tags": ["学习路径", "LLM", "工程", "实验"], "date": "2026-10-20", "desc": "九个阶段串起模型计算、GPU 算子、缓存、推理引擎、服务压测、微调、真实架构和多模态；每阶段都有先修、验收任务与实验，最后完成最小推理引擎。"},
    {"slug": "mini-inference-engine", "title": "从零写一个最小推理引擎", "kind": "deck", "cat": "inference", "tags": ["推理", "KV cache", "调度", "采样", "实验"], "date": "2026-10-20", "desc": "用可运行的 NumPy Transformer 串起正确前向、GQA、RoPE、分页 KV、连续调度、采样和测量；含独立参考检查、三种故障注入、四道习题与实验记录模板。"},
    {"slug": "tensor-layouts", "title": "张量布局、stride 与 GPU 访存", "date": "2026-10-20", "kind": "deck", "cat": "kernels", "tags": ["GPU", "算子优化", "张量布局", "访存"], "desc": "从逻辑索引推导真实地址，拆解 transpose、reshape、合并访存与 shared memory bank；含逐步动画、复制成本计算、排错习题和 NumPy 实验。"},
    {"slug": "gemm-optimization", "title": "GEMM：从三重循环到 Tensor Core", "date": "2026-10-20", "kind": "deck", "cat": "kernels", "tags": ["GPU", "算子优化", "GEMM", "Triton"], "desc": "从矩阵乘法的 FLOPs 与字节账本推导分块复用、shared memory、寄存器与 Tensor Core，含尾块动画、性能诊断、计算题、CPU 实验和完整 Triton 示例。"},
    {"slug": "cuda-streams", "title": "CUDA streams、events 与异步流水", "date": "2026-10-20", "kind": "deck", "cat": "kernels", "tags": ["CUDA", "推理", "性能分析", "并发"], "desc": "从命令队列、跨流依赖与双缓冲到存储寿命、关键路径和正确计时；含可回放时间线、计算题、CPU 调度实验与 CUDA 示例。"},
    {"slug": "torch-compile", "title": "torch.compile：从 Python 到 GPU kernel", "date": "2026-10-20", "kind": "deck", "cat": "kernels", "tags": ["PyTorch", "编译", "推理", "算子优化"], "desc": "从 FX 图、算子融合与 guards 到动态形状、重编译、分桶和冷启动摊销；含内存流量动画、计算题、缓存模型与真实 PyTorch 实验。"},
    {"slug": "rope-long-context", "title": "RoPE 与长上下文扩展", "date": "2026-10-20", "kind": "deck", "cat": "models", "tags": ["RoPE", "长上下文", "位置编码", "KV cache"], "desc": "从二维旋转推导相对位置，逐步计算频率、插值与 YaRN，排查 KV cache 位置和精度问题；含旋转动画、计算题与 NumPy 实验。"},
    {"slug": "sampling-structured-decoding", "title": "采样器与结构化解码", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["采样", "结构化输出", "推理", "投机解码"], "desc": "从 logits、temperature、top-k/top-p 到 token 级语法约束、状态回滚与投机验证，结合概率动画、JSON 栈、计算题和可运行采样实验。"},
    {"slug": "inference-benchmarking", "title": "推理压测与 SLO：测出真正可用的容量", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["推理", "性能分析", "SLO", "排队"], "desc": "从请求时间戳、开放负载与尾延迟到 goodput：用排队动画、容量计算、诊断案例和 CPU 实验建立可复现的压测方法。"},
    {"slug": "prefill-decode-disaggregation", "title": "Prefill / Decode 分离：把计算与缓存交接清楚", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["推理", "并行", "通信", "KV cache"], "desc": "从逐卡数据流与 KV 传输算起，推导排队、资源配比、缓存生命周期和多轮一致性；配计算题与可运行的交接实验。"},
    {"slug": "lora-qlora", "title": "LoRA / QLoRA：参数高效微调", "date": "2026-10-20", "kind": "deck", "cat": "recipes", "tags": ["LoRA", "QLoRA", "微调", "量化", "显存"], "desc": "从低秩更新、梯度与初始化到 NF4、double quant 和完整显存账本；含数值动画、工程诊断、计算题与 NumPy 微调实验。"},
    {"slug": "multi-adapter-serving", "title": "多 Adapter 推理服务", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["LoRA", "推理", "调度", "KV cache", "显存"], "desc": "从共享底座与异构 rank 分组到 adapter 驻留、原子准入、预取、版本和 KV 隔离；含数据流动画、计算题与 CPU 服务实验。"},
    {"slug": "multimodal-inference", "title": "图像与视频推理：token、计算与显存账本", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["多模态", "推理", "KV cache", "视频", "显存"], "desc": "从图像 patch、视频时间网格到视觉编码、语言 KV、调度与缓存，逐项计算多模态推理成本，附动画、计算题和可运行 CPU 实验。"},
    {"slug": "vla-inference", "title": "VLA：从视觉 token 到机器人动作", "date": "2026-10-20", "kind": "deck", "cat": "models", "tags": ["VLA", "多模态", "机器人", "推理", "延迟"], "desc": "从视觉语言输入到动作 token、flow matching 与 action chunk，推导单位、坐标、时延和闭环误差，附动画、习题与纯 CPU 仿真。"},
    {"slug": "deepseek-v4-flash", "title": "DeepSeek-V4-Flash 架构：一个 token 的旅程", "date": "2026-10-20", "kind": "deck", "cat": "models", "tags": ["DeepSeek", "架构", "MoE", "稀疏注意力"], "desc": "从张量形状、mHC、逐通道压缩、因果边界和 MoE 路由深入理解 V4-Flash，含数值动画、工程分析、习题解答与可运行实验。"},
    {"slug": "deepseek-v4-flash-inference", "title": "DeepSeek-V4-Flash 推理：一枚 token，4 张卡", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["DeepSeek", "推理", "KV cache", "并行", "通信"], "desc": "逐卡推导 V4-Flash 推理的权重与缓存容量、FP4 scale、prefill 峰值、FP32 通信和关键路径，含计算题、排错案例与 CPU 实验。"},
    {"slug": "deepseek-v4-pro", "title": "DeepSeek-V4-Pro 架构：一个 token 的旅程", "date": "2026-10-20", "kind": "deck", "cat": "models", "tags": ["DeepSeek", "架构", "MoE", "稀疏注意力"], "desc": "从张量形状、mHC、逐通道压缩、因果边界和 MoE 路由深入理解 V4-Pro，含数值动画、工程分析、习题解答与可运行实验。"},
    {"slug": "deepseek-v4-pro-inference", "title": "DeepSeek-V4-Pro 推理：一枚 token，8 张卡", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["DeepSeek", "推理", "KV cache", "并行", "通信"], "desc": "逐卡推导 V4-Pro 推理的权重与缓存容量、FP4 scale、prefill 峰值、FP32 通信和关键路径，含计算题、排错案例与 CPU 实验。"},
    {"slug": "kv-cache", "title": "KV cache：记住已经算过的历史", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["KV cache", "推理", "显存"], "desc": "从一行 attention 的数值计算出发，推导 KV 为什么可复用、prefill/decode 怎样交接、位置与 mask 如何对齐，再算清 GQA 缓存的容量和读取成本；含可操作图解、符号释义、习题与 NumPy 实验。"},
    {"slug": "kv-cache-compression", "title": "KV cache 压缩：把历史装得更小", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["KV cache", "低精度", "长上下文"], "desc": "推导 GQA 与 MLA 的缓存形状，计算量化误差、分组元数据和残差区预算；附性能诊断、课后习题与可运行压缩实验。"},
    {"slug": "sparse-attention", "title": "Sparse attention：只看需要的历史", "date": "2026-10-20", "kind": "deck", "cat": "models", "tags": ["注意力", "稀疏注意力", "长上下文"], "desc": "从稀疏 softmax 到选择器成本、块利用率、因果索引与 kernel 布局；用数值动画、算术练习和 CPU 实验验证正确性。"},
    {"slug": "linear-attention", "title": "Linear attention：把历史写进状态", "date": "2026-10-20", "kind": "deck", "cat": "models", "tags": ["线性注意力", "架构", "长上下文"], "desc": "推导递推、并行与分块等价，分析状态容量、门控权重与 delta rule 稳定性；附数值动画、计算题和 NumPy 实验。"},
    {"slug": "paged-attention", "title": "PagedAttention：让 KV cache 按需生长", "date": "2026-10-20", "kind": "interactive", "cat": "inference", "tags": ["KV cache", "显存", "推理"], "desc": "从连续缓存的浪费讲到页表寻址、分块 softmax、容量预算和写时复制：用完整推导、三个可操作动态图、四道习题与 CPU 实验理解分页推理。"},
    {"slug": "continuous-batching", "title": "Continuous batching：每一轮重新组队", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["调度", "推理", "KV cache"], "desc": "手算每轮调度、TTFT 和 ITL，理解 chunked prefill、显存准入、公平性与过载；附时间轴动画、习题和离散事件实验。"},
    {"slug": "prefix-caching", "title": "Prefix caching：相同的开头，只算一次", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["KV cache", "推理", "调度"], "desc": "从哈希链与满块边界，到计算身份、引用驱逐、显存收益和延迟模型；附数值演算、三道练习与缓存模拟代码。"},
    {"slug": "speculative-decoding", "title": "投机解码", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["投机解码", "推理"], "desc": "从概率质量守恒推导投机采样，手算接受长度与加速比，排查 logits/KV 对齐；含数值动画、4 道练习和可运行 CPU 实验。"},
    {"slug": "dspark", "title": "DSpark", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["投机解码", "DeepSeek", "调度"], "desc": "深入 DSpark 的 Markov 张量、接受率训练、置信度校准与因果预算调度；手算吞吐和选择偏差，配 3 道练习及 CPU 模拟实验。"},
    {"slug": "llm-quantization", "title": "LLM 量化", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["量化", "低精度", "显存"], "desc": "从 INT4 编码与元数据算到量化 kernel、GPTQ 补偿、AWQ 和 SmoothQuant；含显存/带宽账本、4 道计算练习及可运行实验。"},
    {"slug": "distributed-inference", "title": "分布式推理：数据怎样走过每张卡", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["推理", "并行", "通信", "MoE"], "desc": "从张量形状和手算 MLP 推导 TP、PP、EP，计算每卡显存、ring 通信与流水线空泡，用诊断案例、习题和 Python 实验分析部署方案。"},
    {"slug": "cpu-overhead", "title": "CPU overhead：GPU 为什么在等", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["推理", "调度", "profiling"], "desc": "用异步队列递推、CUDA 计时、Nsight trace 和 Amdahl 定律定位 CPU 开销，分析吞吐与尾延迟，配计算题和可运行的提交队列实验。"},
    {"slug": "cuda-graph", "title": "CUDA Graph：把一轮计算录下来", "date": "2026-10-20", "kind": "deck", "cat": "kernels", "tags": ["GPU", "kernel", "推理"], "desc": "从固定存储与输出别名到真实 PyTorch capture/replay，计算形状桶、显存预算和回本次数，配习题、CPU 模拟及可下载 CUDA 实验。"},
    {"slug": "flash-attention", "title": "FlashAttention 1、2、3", "date": "2026-10-20", "kind": "deck", "cat": "kernels", "tags": ["GPU", "kernel", "注意力"], "desc": "推导 online softmax 与 attention 反向，演算 tile 资源、并行与流水线成本；含 PyTorch 用法、排查案例、三道习题和可运行的 NumPy 前反向实验。"},
    {"slug": "rmsnorm-kernel", "title": "RMSNorm 与 GPU 算子优化", "date": "2026-10-19", "kind": "deck", "cat": "kernels", "tags": ["GPU", "kernel", "RMSNorm"], "desc": "从逐项数值与反向推导到 CUDA/Triton 归约、向量化、寄存器与带宽账本；加入精度排查、计算题、梯度差分验证和可运行 CPU 实验。"},
    {"slug": "gradient-checkpointing", "title": "Gradient checkpointing", "date": "2026-10-19", "kind": "deck", "cat": "training", "tags": ["显存", "激活重算"], "desc": "从激活生命周期推导分段重算的显存和时间成本，分析随机数、non-reentrant、FSDP 与排查；含计算题、梯度验证和 NumPy/PyTorch 动手实验。"},
    {"slug": "optimizers", "title": "优化器：从 SGD 到 Muon", "date": "2026-10-19", "kind": "deck", "cat": "recipes", "tags": ["优化器"], "desc": "一套动画幻灯片：把 GD、动量、Nesterov、AdaGrad、RMSProp、Adam、AdamW、Lion、Adafactor、Shampoo 和 Muon 放到同一片 loss 地形上实际跑一遍，附可交互的优化器游乐场。"},
    {"slug": "deepseek-v4", "title": "DeepSeek-V4 技术报告精读", "date": "2026-10-19", "kind": "interactive", "cat": "models", "tags": ["DeepSeek", "架构", "稀疏注意力", "MoE", "长上下文", "优化器", "低精度"], "desc": "拆解 DeepSeek-V4：CSA/HCA 注意力、mHC、Muon、FP4 QAT 与多教师蒸馏，附成本计算器。"},
    {"slug": "qwen3-next", "title": "Qwen3-Next：混合线性注意力与超稀疏 MoE", "date": "2026-10-19", "kind": "interactive", "cat": "models", "tags": ["Qwen", "架构", "线性注意力", "MoE", "投机解码"], "desc": "从线性注意力推到 Gated DeltaNet、门控注意力、3:1 混合、512 专家 MoE 与 MTP。"},
    {"slug": "kimi-k3", "title": "Kimi K3：KDA、注意力残差与 Stable LatentMoE", "date": "2026-10-19", "kind": "interactive", "cat": "models", "tags": ["Kimi", "架构", "线性注意力", "MoE", "优化器", "长上下文"], "desc": "精读 Kimi K3：有下界的 KDA、注意力残差、Stable LatentMoE 与分位数均衡，以及 1M 上下文系统。"},
    {"slug": "kimi-k2", "title": "Kimi K2：MuonClip 与万亿参数 Agent 模型", "date": "2026-10-19", "kind": "interactive", "cat": "models", "tags": ["Kimi", "架构", "优化器", "MoE", "RL"], "desc": "精读 Kimi K2：Muon 与 QK-Clip 推导、数据改写、超稀疏 MoE、训练基础设施、Agent 数据合成与 RL。"},
    {"slug": "kimi-linear", "title": "Kimi Linear：Kimi Delta Attention", "date": "2026-10-19", "kind": "interactive", "cat": "models", "tags": ["Kimi", "线性注意力", "架构", "KV cache"], "desc": "从线性注意力、DeltaNet 到 KDA：通道级门控、分块并行算法、3:1 混合与 NoPE、效率实验。"},
    {"slug": "smol-decisions", "title": "从目标到决策：训练什么、怎么做消融", "date": "2026-10-19", "kind": "interactive", "cat": "recipes", "tags": ["消融", "评测"], "desc": "该不该训、训什么、怎样用小规模消融做决策，以及评测噪声与消融成本。"},
    {"slug": "smol-architecture", "title": "架构选择：注意力、位置编码、MoE 与混合模型", "date": "2026-10-19", "kind": "interactive", "cat": "recipes", "tags": ["架构", "注意力", "MoE", "KV cache"], "desc": "MHA/GQA/MLA 与 KV cache、文档掩码、RoPE/NoPE、稳定性技巧、MoE 设计与混合模型。"},
    {"slug": "smol-optim-data", "title": "分词器、优化器、超参与数据配比", "date": "2026-10-19", "kind": "interactive", "cat": "recipes", "tags": ["优化器", "数据"], "desc": "BPE 与分词器评估、AdamW 与 Muon、学习率调度、batch size 与超参 scaling law、数据配比。"},
    {"slug": "smol-marathon", "title": "训练马拉松：吞吐、loss spike 与中途调整", "date": "2026-10-19", "kind": "interactive", "cat": "recipes", "tags": ["稳定性", "长上下文"], "desc": "SmolLM3 长跑实录：起跑清单、四个谜案、loss spike 诊断、多阶段 mid-training 与长上下文扩展。"},
    {"slug": "smol-post-training", "title": "后训练：SFT、偏好优化与 RL", "date": "2026-10-19", "kind": "interactive", "cat": "recipes", "tags": ["后训练", "RL"], "desc": "后训练全流程：评测先行、chat template 与掩码、SFT 调参、推理 mid-training、DPO/APO 推导、GRPO 与长度惩罚。"},
    {"slug": "smol-infrastructure", "title": "基础设施：GPU 集群、通信与存储", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["硬件", "通信", "GPU"], "desc": "H100 内部、PCIe/NVLink/EFA 实测带宽、nccl-tests、存储分层、节点健康与 checkpoint，以及该用多少卡。"},
    {"slug": "scaling-memory", "title": "训练的账本：显存、FLOPs 与 MFU", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["显存"], "desc": "一步训练里显存花在哪、FLOPs 怎么数、MFU 怎么算：从 16 字节/参数推到 6ND，附显存计算器与训练时长估算器。"},
    {"slug": "scaling-roofline", "title": "Roofline：算力、带宽与算术强度", "date": "2026-10-19", "kind": "interactive", "cat": "kernels", "tags": ["硬件", "GPU"], "desc": "一个运算到底卡在算力、显存带宽还是网络上：算术强度、roofline 图、矩阵乘法的临界 batch，以及 TPU 与 GPU 的硬件结构。"},
    {"slug": "scaling-collectives", "title": "集合通信与分片矩阵乘法", "date": "2026-10-19", "kind": "deck", "cat": "training", "tags": ["通信", "并行"], "desc": "从 broadcast 到 all-to-all，推导环形集合通信的代价，再用分片记号讲清分片矩阵乘法。"},
    {"slug": "scaling-data-parallel", "title": "数据并行、ZeRO 与 FSDP", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["并行", "显存", "通信"], "desc": "从朴素数据并行讲到 ZeRO-1/2/3 与 FSDP：通信重叠、分桶、no_sync、显存公式与通信瓶颈。"},
    {"slug": "scaling-tensor-parallel", "title": "张量并行与序列并行", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["并行", "通信"], "desc": "从矩阵乘法的两种切法讲起：Megatron 张量并行、f/g 算子、词表并行、序列并行与通信量。"},
    {"slug": "scaling-context-parallel", "title": "上下文并行与 Ring Attention", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["并行", "长上下文", "注意力"], "desc": "长序列训练怎么切激活：online softmax 合并、Ring Attention、zig-zag 负载均衡与 Ulysses。"},
    {"slug": "scaling-pipeline", "title": "流水线并行：从 GPipe 到 DualPipe", "date": "2026-10-19", "kind": "deck", "cat": "training", "tags": ["并行"], "desc": "交互幻灯片：推导 AFAB、1F1B、交错、Zero Bubble 与 DualPipe 的气泡与显存，附调度模拟器。"},
    {"slug": "scaling-moe", "title": "MoE 与专家并行", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["MoE", "并行", "通信"], "desc": "从路由器、容量、负载均衡损失到无辅助损失偏置，再到专家并行的 all-to-all 与通信成本。"},
    {"slug": "scaling-5d", "title": "5D 并行：组合与选型", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["并行"], "desc": "DP、TP、CP、PP、EP 各切什么、通信放在网络哪层、怎么组合，附配置规划器与设备网格演示。"},
    {"slug": "scaling-jax", "title": "JAX 分片编程与 Profiling", "date": "2026-10-19", "kind": "interactive", "cat": "training", "tags": ["JAX", "并行", "profiling"], "desc": "用 mesh、PartitionSpec、shard_map 写分片程序，读懂 HLO 与 profile，分清计算、通信与空闲。"},
    {"slug": "scaling-gpu-kernels", "title": "GPU 内核与 FlashAttention", "date": "2026-10-19", "kind": "interactive", "cat": "kernels", "tags": ["GPU", "kernel", "注意力"], "desc": "从 SM、warp、内存层级讲起，到访存合并、分块、融合、Triton，再到 FlashAttention 1–3 的推导与实现。"},
    {"slug": "scaling-mixed-precision", "title": "混合精度与 FP8 训练", "date": "2026-10-19", "kind": "interactive", "cat": "kernels", "tags": ["低精度", "量化"], "desc": "浮点格式、舍入、损失缩放、FP32 主权重，到 FP8 的缩放策略、DeepSeek-V3 的细粒度量化与 MX/NVFP4 微缩放。"},
    {"slug": "scaling-inference", "title": "Transformer 推理的数学", "date": "2026-10-19", "kind": "interactive", "cat": "inference", "tags": ["推理", "KV cache", "投机解码", "并行"], "desc": "从 KV cache 到投机解码：推导带宽与算力瓶颈、延迟下界、吞吐权衡和多卡服务账本；公式旁解释符号与单位，并提供可悬停、点击的变量释义。"},
    {"slug": "next-token", "title": "一次只猜一个 token", "date": "2026-10-19", "kind": "deck", "cat": "basics", "tags": ["推理"], "desc": "用 8 页动画讲清楚大模型生成文本的循环：切词、打分、采样、再来一遍。"},
    {"slug": "hello-world", "title": "新博客，从零开始", "date": "2026-10-19", "kind": "interactive", "cat": "basics", "tags": [], "desc": "这个博客怎么搭的，以及一个可以拖动的 temperature 采样演示。"},
  ];

  window.SITE = { POSTS, CATEGORIES, KINDS };

  // ---------- post pages: tags under the header, related posts at the end ----------
  const slug = location.pathname.match(/^\/posts\/([^/]+)\//)?.[1];
  const me = POSTS.find((p) => p.slug === slug);
  if (!me) return;
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const tagLink = (t) => '<a class="tag" href="/blog/?tag=' + encodeURIComponent(t) + '">#' + esc(t) + '</a>';

  function render() {
    const meta = document.querySelector('.post-header .post-meta');
    if (meta && !meta.querySelector('.post-tax')) {
      const span = document.createElement('span');
      span.className = 'post-tax';
      span.innerHTML = '<a class="cat" href="/blog/?cat=' + me.cat + '">' + esc(CATEGORIES[me.cat]) + '</a>' + me.tags.map(tagLink).join('');
      meta.appendChild(span);
    }
    const main = document.querySelector('main.wrap');
    if (!main || document.body.classList.contains('deck-page')) return;
    const links = window.POST_LINKS || {};
    const by = (s) => POSTS.find((p) => p.slug === s);
    const out = (links[slug] || []).map(by).filter(Boolean);
    const inn = Object.keys(links).filter((s) => s !== slug && links[s].includes(slug)).map(by).filter(Boolean);
    const seen = new Set([slug, ...out.map((p) => p.slug), ...inn.map((p) => p.slug)]);
    const near = POSTS.filter((p) => !seen.has(p.slug))
      .map((p) => ({ p, n: p.tags.filter((t) => me.tags.includes(t)).length + (p.cat === me.cat ? 0.5 : 0) }))
      .filter((x) => x.n >= 1).sort((a, b) => b.n - a.n).slice(0, 4).map((x) => x.p);
    const item = (p) => '<li><a href="/posts/' + p.slug + '/">' + esc(p.title) + '</a> <span class="' + KINDS[p.kind][1] + '">' + KINDS[p.kind][0] + '</span></li>';
    const group = (title, list) => (list.length ? '<div><h3>' + title + '</h3><ul>' + list.map(item).join('') + '</ul></div>' : '');
    const html = group('本文引用', out) + group('引用本文', inn) + group('相关主题', near);
    if (!html) return;
    const box = document.createElement('section');
    box.className = 'related';
    box.innerHTML = '<h2>相关文章</h2><div class="related-grid">' + html + '</div>';
    main.appendChild(box);
  }
  // Hover / focus a link to another post: show its title, kind and summary.
  function previews() {
    const card = document.createElement('div');
    card.className = 'link-card';
    card.hidden = true;
    document.body.appendChild(card);
    let hideTimer = 0;
    document.querySelectorAll('.prose a[href^="/posts/"]').forEach((a) => {
      const s = a.getAttribute('href').split('/')[2];
      const p = POSTS.find((x) => x.slug === s);
      if (!p || s === slug) return;
      const show = () => {
        clearTimeout(hideTimer);
        card.innerHTML = '<span class="' + KINDS[p.kind][1] + '">' + KINDS[p.kind][0] + '</span> <span class="lc-cat">' + esc(CATEGORIES[p.cat]) + '</span>' +
          '<b>' + esc(p.title) + '</b><span class="lc-desc">' + esc(p.desc) + '</span>';
        card.hidden = false;
        const r = a.getBoundingClientRect();
        const w = Math.min(340, innerWidth - 24);
        card.style.width = w + 'px';
        card.style.left = Math.max(12, Math.min(innerWidth - w - 12, r.left)) + 'px';
        const below = r.bottom + 8;
        card.style.top = (below + card.offsetHeight > innerHeight ? r.top - card.offsetHeight - 8 : below) + 'px';
      };
      const hide = () => { hideTimer = setTimeout(() => { card.hidden = true; }, 120); };
      a.addEventListener('mouseenter', show);
      a.addEventListener('focus', show);
      a.addEventListener('mouseleave', hide);
      a.addEventListener('blur', hide);
    });
    addEventListener('scroll', () => { card.hidden = true; }, { passive: true });
  }

  const start = () => { render(); if (matchMedia('(hover: hover)').matches) previews(); };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
