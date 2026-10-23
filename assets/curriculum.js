// Shared learning map. Post titles and descriptions come from assets/posts.js.
(() => {
  const stages = [
    {
      id: 'foundations', title: '把一次前向算清楚',
      goal: '写出张量形状、位置和因果关系，能解释完整前向与增量前向为什么一致。',
      prerequisite: '会读 Python；知道向量、矩阵乘法与概率归一化。',
      check: '手算一行 attention；画出转置后的地址；验证不同长度的 dense / cached logits。',
      core: ['next-token', 'scaling-inference', 'tensor-layouts', 'rope-long-context', 'kv-cache'],
      extra: ['smol-architecture'],
    },
    {
      id: 'kernels', title: '理解 GPU 在忙什么',
      goal: '从 FLOPs、字节和依赖关系判断瓶颈，读懂 kernel 与 CPU 的时间线。',
      prerequisite: '阶段 1 的张量形状与地址；矩阵乘法的运算量。',
      check: '算 GEMM 的算术强度；给异步流水补 event；区分编译冷启动与稳定运行。',
      core: ['scaling-roofline', 'gemm-optimization', 'cuda-streams', 'rmsnorm-kernel', 'flash-attention', 'torch-compile', 'cpu-overhead', 'cuda-graph'],
      extra: ['scaling-gpu-kernels', 'scaling-jax'],
    },
    {
      id: 'memory', title: '管理历史与模型容量',
      goal: '把历史状态、权重和激活的压缩连到实际存储格式，算出容量、误差与访存成本。',
      prerequisite: '阶段 1 的 KV 与位置；阶段 2 的带宽和计算预算。',
      check: '核对 KV 与递归状态容量；验证分块和逐步计算一致；比较剪枝、量化与激活筛选的误差和实际访存。',
      core: ['paged-attention', 'kv-cache-compression', 'sparse-attention', 'linear-attention', 'state-space-models', 'llm-quantization', 'weight-pruning', 'activation-sparsity'],
      extra: ['kimi-linear', 'qwen3-next'],
    },
    {
      id: 'engine', title: '把模型接成推理引擎',
      goal: '跟踪请求、缓存与采样器的状态，让不同长度的请求共同推进。',
      prerequisite: '阶段 1 的因果前向与阶段 3 的分页缓存。',
      check: '画出一轮调度；说明最后一个采样 token 是否已进入 KV；验证采样频率和投机回滚。',
      core: ['continuous-batching', 'prefix-caching', 'sampling-structured-decoding', 'speculative-decoding', 'dspark'],
      extra: [],
    },
    {
      id: 'serving', title: '用测量选择服务方案',
      goal: '将延迟、吞吐、排队、通信与容量放在同一份账本里。',
      prerequisite: '阶段 2 的时间线、阶段 3 的容量、阶段 4 的请求生命周期。',
      check: '设计开放环压测；计算满足 SLO 的 goodput；在相同硬件预算下比较合并与 P/D 分离。',
      core: ['distributed-inference', 'inference-benchmarking', 'prefill-decode-disaggregation'],
      extra: ['scaling-collectives', 'scaling-tensor-parallel', 'scaling-context-parallel', 'scaling-pipeline', 'scaling-moe', 'scaling-5d'],
    },
    {
      id: 'pretraining', title: '设计预训练算法与实验',
      goal: '把预算、数据、学习目标与更新尺度连起来，再比较低精度、稀疏连接和 dense 转 MoE 的训练选择。',
      prerequisite: '阶段 1 的概率与矩阵计算；链式法则和 SGD / Adam（可先读《优化器：从 SGD 到 Muon》）。',
      check: '分配预算、筛选与混合数据；迁移宽度超参；跟踪三个 GEMM 的数值误差；检查双向 2:4 掩码与 upcycling 输出尺度。',
      core: ['pretraining-scaling', 'pretraining-data-curation', 'pretraining-data-mixture', 'pretraining-mup', 'pretraining-objectives', 'pretraining-stability', 'scaling-mixed-precision', 'sparse-pretraining', 'moe-upcycling'],
      extra: ['scaling-moe', 'smol-decisions', 'smol-optim-data', 'smol-marathon'],
    },
    {
      id: 'adaptation', title: '微调、模型复用与多租户',
      goal: '理解梯度与优化器状态，比较微调与合并的能力变化，再把不同 adapter 放进同一服务。',
      prerequisite: '阶段 1 的矩阵形状与链式法则（优化器课会补齐）；进入多 Adapter 服务前完成阶段 4。',
      check: '核对 LoRA 梯度和训练显存；对齐合并模型的参数与词表；按任务评估收益；证明 adapter 变化不会复用旧 KV。',
      core: ['scaling-memory', 'optimizers', 'gradient-checkpointing', 'lora-qlora', 'model-merging', 'multi-adapter-serving'],
      extra: ['scaling-data-parallel', 'smol-decisions', 'smol-optim-data', 'smol-marathon', 'smol-post-training', 'smol-infrastructure'],
    },
    {
      id: 'architectures', title: '设计信息流并拆开真实模型',
      goal: '比较跨层连接、共享深度、查表记忆和概念计算，再把架构图转换成张量、状态与通信。',
      prerequisite: '残差分支需要矩阵乘法、softmax 与链式法则；真实模型推理需要阶段 3 的缓存和阶段 5 的通信。',
      check: '手算多路残差与深度加权；做删层和递归计算实验；核对记忆查询及概念压缩成本；重建真实模型的跨卡关键路径。',
      core: ['hyper-connections', 'manifold-hyper-connections', 'attention-residuals', 'effective-depth', 'recurrent-depth', 'over-encoding', 'engram', 'concept-models', 'deepseek-v4-flash', 'deepseek-v4-flash-inference', 'deepseek-v4-pro', 'deepseek-v4-pro-inference'],
      extra: ['deepseek-v4', 'kimi-k2', 'kimi-k3', 'iquest-q1'],
    },
    {
      id: 'multimodal', title: '从图像和视频走向动作',
      goal: '将分辨率、帧率与动作频率转换成 token、显存和端到端时延。',
      prerequisite: '阶段 1 的位置与 KV、阶段 4 的生成、阶段 5 的延迟预算。',
      check: '给视频请求列出 encoder / LLM 两份账；计算 action chunk 的闭环延迟与观测年龄。',
      core: ['multimodal-inference', 'vla-inference'],
      extra: [],
    },
    {
      id: 'capstone', title: '综合实战：最小推理引擎',
      goal: '完成正确前向、分页 KV、连续调度、采样、测量与故障定位。',
      prerequisite: '阶段 1、3、4、5 的主线；优化实验再使用阶段 2。训练与多模态可作为后续分支。',
      check: '提交可复现命令、正确性证据、容量表、实验记录和三种故障的定位过程。',
      core: ['mini-inference-engine'], extra: [],
    },
  ];
  window.CURRICULUM = { title: 'LLM 工程学习路径', stages };
  const parts = location.pathname.split('/').filter(Boolean);
  if (parts[0] !== 'posts') return;
  const slug = parts[1];
  const stage = stages.find(s => [...s.core, ...s.extra].includes(slug));
  if (!stage) return;
  const route = '/posts/llm-engineering/#' + stage.id;
  const header = document.querySelector('.post-header');
  if (!header) return;
  const p = document.createElement('p');
  p.className = 'post-meta course-context';
  const a = document.createElement('a');
  a.href = route;
  a.textContent = '学习路径 · ' + stage.title;
  p.append(a);
  header.append(p);
})();
