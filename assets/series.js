// Series navigation for multi-part posts. A post opts in with
//
//   <nav class="series" data-series="scaling"></nav>        (under the header)
//   <nav class="series-pager" data-series="scaling"></nav>  (after the article)
//   <script src="/assets/series.js"></script>
//
// and this file fills both from the list below. The current post is found
// from the URL. To add a post to a series, add it here; titles should match
// the post's <h1>.
(() => {
  const SERIES = {
    architectures: {
      title: "模型结构：状态、深度与条件记忆",
      posts: [
        ["state-space-models", "状态空间模型怎样记住历史：从 S4 到 Mamba-3"],
        ["recurrent-depth", "同一组参数多算几遍：共享深度与递归推理"],
        ["over-encoding", "Over-Encoding：用查表扩展语言模型容量"],
        ["engram", "Engram：条件记忆怎样与 MoE 分工"],
        ["concept-models", "从 token 到概念：句向量生成与动态分段"],
      ],
    },
    sparsity: {
      title: "稀疏计算：权重、训练与激活",
      posts: [
        ["weight-pruning", "权重剪枝：从幅值到数据感知重建"],
        ["sparse-pretraining", "稀疏预训练：动态连接怎样变成真实加速"],
        ["activation-sparsity", "激活稀疏：当前 token 需要哪些神经元"],
      ],
    },
    pretraining: {
      title: "预训练算法：从预算与数据到学习动态",
      posts: [
        ["pretraining-scaling", "训练预算怎样分配：Scaling Laws 与模型、数据的取舍"],
        ["pretraining-data-curation", "预训练数据怎样筛选：去重、质量与多样性"],
        ["pretraining-data-mixture", "数据配比怎样学出来：DoReMi、DoGE 与 RegMix"],
        ["pretraining-mup", "从小模型调参到大模型：μP 与超参数迁移"],
        ["pretraining-objectives", "一段文本提供什么监督：从去噪、FIM 到多 token 预测"],
        ["pretraining-stability", "深层 Transformer 怎样稳定训练"],
        ["moe-upcycling", "已有模型怎样长成 MoE：从权重复用到专家分化"],
      ],
    },
    hyperconnections: {
      title: "Hyper-Connections 与深层信息流",
      posts: [
        ["hyper-connections", "Hyper-Connections：从一条残差流到多路信息交换"],
        ["manifold-hyper-connections", "mHC：用双随机约束稳定多路残差"],
        ["attention-residuals", "Attention Residuals：沿深度选择信息"],
        ["effective-depth", "一个模型到底用了多少深度：测量、剪层与提前退出"],
      ],
    },
    scaling: {
      title: '大模型规模化训练',
      posts: [
        ['scaling-memory', '训练的账本：显存、FLOPs 与 MFU'],
        ['scaling-roofline', 'Roofline：算力、带宽与算术强度'],
        ['scaling-collectives', '集合通信与分片矩阵乘法'],
        ['scaling-data-parallel', '数据并行、ZeRO 与 FSDP'],
        ['scaling-tensor-parallel', '张量并行与序列并行'],
        ['scaling-context-parallel', '上下文并行与 Ring Attention'],
        ['scaling-pipeline', '流水线并行：从 GPipe 到 DualPipe'],
        ['scaling-moe', 'MoE 与专家并行'],
        ['scaling-5d', '5D 并行：组合与选型'],
        ['scaling-jax', 'JAX 分片编程与 Profiling'],
        ['scaling-gpu-kernels', 'GPU 内核与 FlashAttention'],
        ["scaling-mixed-precision", "混合精度训练：从 BF16 到 FP8 与 NVFP4"],
        ['scaling-inference', 'Transformer 推理的数学'],
      ],
    },
    smol: {
      title: '从零训练一个 LLM',
      posts: [
        ['smol-decisions', '从目标到决策：训练什么、怎么做消融'],
        ['smol-architecture', '架构选择：注意力、位置编码、MoE 与混合模型'],
        ['smol-optim-data', '分词器、优化器、超参与数据配比'],
        ['smol-marathon', '训练马拉松：吞吐、loss spike 与中途调整'],
        ['smol-post-training', '后训练：SFT、偏好优化与 RL'],
        ['smol-infrastructure', '基础设施：GPU 集群、通信与存储'],
      ],
    },
    systems: {
      "title": "LLM 工程：从张量到推理服务",
      "posts": [
        [
          "llm-engineering",
          "LLM 工程学习路径：从张量到推理服务"
        ],
        [
          "next-token",
          "一次只猜一个 token"
        ],
        [
          "scaling-inference",
          "Transformer 推理的数学"
        ],
        [
          "tensor-layouts",
          "张量布局、stride 与 GPU 访存"
        ],
        [
          "rope-long-context",
          "RoPE 与长上下文扩展"
        ],
        [
          "kv-cache",
          "KV cache：记住已经算过的历史"
        ],
        [
          "scaling-roofline",
          "Roofline：算力、带宽与算术强度"
        ],
        [
          "gemm-optimization",
          "GEMM：从三重循环到 Tensor Core"
        ],
        [
          "cuda-streams",
          "CUDA streams、events 与异步流水"
        ],
        [
          "rmsnorm-kernel",
          "RMSNorm 与 GPU 算子优化"
        ],
        [
          "flash-attention",
          "FlashAttention 1、2、3"
        ],
        [
          "torch-compile",
          "torch.compile：从 Python 到 GPU kernel"
        ],
        [
          "cpu-overhead",
          "CPU overhead：GPU 为什么在等"
        ],
        [
          "cuda-graph",
          "CUDA Graph：把一轮计算录下来"
        ],
        [
          "paged-attention",
          "PagedAttention：让 KV cache 按需生长"
        ],
        [
          "kv-cache-compression",
          "KV cache 压缩：把历史装得更小"
        ],
        [
          "sparse-attention",
          "Sparse attention：只看需要的历史"
        ],
        [
          "linear-attention",
          "Linear attention：把历史写进状态"
        ],
        [
          "state-space-models",
          "状态空间模型怎样记住历史：从 S4 到 Mamba-3"
        ],
        [
          "llm-quantization",
          "LLM 量化"
        ],
        [
          "weight-pruning",
          "权重剪枝：从幅值到数据感知重建"
        ],
        [
          "activation-sparsity",
          "激活稀疏：当前 token 需要哪些神经元"
        ],
        [
          "continuous-batching",
          "Continuous batching：每一轮重新组队"
        ],
        [
          "prefix-caching",
          "Prefix caching：相同的开头，只算一次"
        ],
        [
          "sampling-structured-decoding",
          "采样器与结构化解码"
        ],
        [
          "speculative-decoding",
          "投机解码"
        ],
        [
          "dspark",
          "DSpark"
        ],
        [
          "distributed-inference",
          "分布式推理：数据怎样走过每张卡"
        ],
        [
          "inference-benchmarking",
          "推理压测与 SLO：测出真正可用的容量"
        ],
        [
          "prefill-decode-disaggregation",
          "Prefill / Decode 分离：把计算与缓存交接清楚"
        ],
        [
          "pretraining-scaling",
          "训练预算怎样分配：Scaling Laws 与模型、数据的取舍"
        ],
        [
          "pretraining-data-curation",
          "预训练数据怎样筛选：去重、质量与多样性"
        ],
        [
          "pretraining-data-mixture",
          "数据配比怎样学出来：DoReMi、DoGE 与 RegMix"
        ],
        [
          "pretraining-mup",
          "从小模型调参到大模型：μP 与超参数迁移"
        ],
        [
          "pretraining-objectives",
          "一段文本提供什么监督：从去噪、FIM 到多 token 预测"
        ],
        [
          "pretraining-stability",
          "深层 Transformer 怎样稳定训练"
        ],
        [
          "scaling-mixed-precision",
          "混合精度训练：从 BF16 到 FP8 与 NVFP4"
        ],
        [
          "sparse-pretraining",
          "稀疏预训练：动态连接怎样变成真实加速"
        ],
        [
          "moe-upcycling",
          "已有模型怎样长成 MoE：从权重复用到专家分化"
        ],
        [
          "scaling-memory",
          "训练的账本：显存、FLOPs 与 MFU"
        ],
        [
          "optimizers",
          "优化器：从 SGD 到 Muon"
        ],
        [
          "gradient-checkpointing",
          "Gradient checkpointing"
        ],
        [
          "lora-qlora",
          "LoRA / QLoRA：参数高效微调"
        ],
        [
          "model-merging",
          "多个 checkpoint 怎样合成一个模型：从 Model Soups 到参数对齐"
        ],
        [
          "multi-adapter-serving",
          "多 Adapter 推理服务"
        ],
        [
          "hyper-connections",
          "Hyper-Connections：从一条残差流到多路信息交换"
        ],
        [
          "manifold-hyper-connections",
          "mHC：用双随机约束稳定多路残差"
        ],
        [
          "attention-residuals",
          "Attention Residuals：沿深度选择信息"
        ],
        [
          "effective-depth",
          "一个模型到底用了多少深度：测量、剪层与提前退出"
        ],
        [
          "recurrent-depth",
          "同一组参数多算几遍：共享深度与递归推理"
        ],
        [
          "over-encoding",
          "Over-Encoding：用查表扩展语言模型容量"
        ],
        [
          "engram",
          "Engram：条件记忆怎样与 MoE 分工"
        ],
        [
          "concept-models",
          "从 token 到概念：句向量生成与动态分段"
        ],
        [
          "deepseek-v4-flash",
          "DeepSeek-V4-Flash 架构：一个 token 的旅程"
        ],
        [
          "deepseek-v4-flash-inference",
          "DeepSeek-V4-Flash 推理：一枚 token，4 张卡"
        ],
        [
          "deepseek-v4-pro",
          "DeepSeek-V4-Pro 架构：一个 token 的旅程"
        ],
        [
          "deepseek-v4-pro-inference",
          "DeepSeek-V4-Pro 推理：一枚 token，8 张卡"
        ],
        [
          "multimodal-inference",
          "图像与视频推理：token、计算与显存账本"
        ],
        [
          "vla-inference",
          "VLA：从视觉 token 到机器人动作"
        ],
        [
          "mini-inference-engine",
          "从零写一个最小推理引擎"
        ]
      ]
    },
    reports: {
      title: '模型技术报告精读',
      posts: [
        ['iquest-q1', 'IQuest-Q1：Stable SplitNorm、混合注意力与递归 MTP'],
        ['deepseek-v4', 'DeepSeek-V4 技术报告精读'],
        ['qwen3-next', 'Qwen3-Next：混合线性注意力与超稀疏 MoE'],
        ['kimi-k2', 'Kimi K2：MuonClip 与万亿参数 Agent 模型'],
        ['kimi-linear', 'Kimi Linear：Kimi Delta Attention'],
        ['kimi-k3', 'Kimi K3：KDA、注意力残差与 Stable LatentMoE'],
      ],
    },
  };

  window.SERIES = SERIES;

  const slug = location.pathname.split('/').filter(Boolean).pop();
  const link = (s) => '/posts/' + s + '/';
  const esc = (t) => t.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  document.querySelectorAll('.series[data-series]').forEach((el) => {
    const s = SERIES[el.dataset.series];
    if (!s) return;
    const i = s.posts.findIndex(([p]) => p === slug);
    const items = s.posts.map(([p, t], k) =>
      '<li' + (k === i ? ' aria-current="page"' : '') + '><a href="' + link(p) + '">' + esc(t) + '</a></li>').join('');
    el.innerHTML = '<details><summary>系列 · <b>' + esc(s.title) + '</b>' +
      (i >= 0 ? '（第 ' + (i + 1) + ' / ' + s.posts.length + ' 篇）' : '') +
      '</summary><ol>' + items + '</ol></details>';
  });

  document.querySelectorAll('.series-pager[data-series]').forEach((el) => {
    const s = SERIES[el.dataset.series];
    if (!s) return;
    const i = s.posts.findIndex(([p]) => p === slug);
    if (i < 0) return;
    const prev = s.posts[i - 1];
    const next = s.posts[i + 1];
    el.innerHTML =
      (prev ? '<a class="prev" href="' + link(prev[0]) + '"><small>← 上一篇</small>' + esc(prev[1]) + '</a>' : '') +
      (next ? '<a class="next" href="' + link(next[0]) + '"><small>下一篇 →</small>' + esc(next[1]) + '</a>' : '');
  });
})();
