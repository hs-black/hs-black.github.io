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
    {"slug": "deepseek-v4-flash", "title": "DeepSeek-V4-Flash 架构：一个 token 的旅程", "date": "2026-10-20", "kind": "deck", "cat": "models", "tags": ["DeepSeek", "架构", "MoE", "稀疏注意力"], "desc": "用可回放动画讲清 V4-Flash 的 mHC、SWA、CSA、HCA、MoE 和输出头。"},
    {"slug": "deepseek-v4-flash-inference", "title": "DeepSeek-V4-Flash 推理：一枚 token，4 张卡", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["DeepSeek", "推理", "KV cache", "并行", "通信"], "desc": "逐卡动画展示 V4-Flash 的 HBM 读取、注意力分片、MoE 通信、KV 更新与 prefill/decode。"},
    {"slug": "deepseek-v4-pro", "title": "DeepSeek-V4-Pro 架构：一个 token 的旅程", "date": "2026-10-20", "kind": "deck", "cat": "models", "tags": ["DeepSeek", "架构", "MoE", "稀疏注意力"], "desc": "用可回放动画讲清 V4-Pro 的 mHC、SWA、CSA、HCA、MoE 和输出头。"},
    {"slug": "deepseek-v4-pro-inference", "title": "DeepSeek-V4-Pro 推理：一枚 token，8 张卡", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["DeepSeek", "推理", "KV cache", "并行", "通信"], "desc": "逐卡动画展示 V4-Pro 的 HBM 读取、注意力分片、MoE 通信、KV 更新与 prefill/decode。"},
    {"slug": "kv-cache", "title": "KV cache：记住已经算过的历史", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["KV cache", "推理", "显存"], "desc": "用可回放动画看清 QKV、因果掩码、prefill 写入、decode 读取，以及 KV cache 的容量与带宽成本。"},
    {"slug": "kv-cache-compression", "title": "KV cache 压缩：把历史装得更小", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["KV cache", "低精度", "长上下文"], "desc": "用动画区分 GQA/MQA、MLA latent、token 合并与淘汰、低位量化，计算容量收益并理解精度与带宽代价。"},
    {"slug": "sparse-attention", "title": "Sparse attention：只看需要的历史", "date": "2026-10-20", "kind": "deck", "cat": "models", "tags": ["注意力", "稀疏注意力", "长上下文"], "desc": "动画讲解 causal mask、滑动窗口、全局 token、内容选择与 block sparse，理解少读哪些 KV、选择成本，以及稀疏访问和缓存压缩的区别。"},
    {"slug": "linear-attention", "title": "Linear attention：把历史写进状态", "date": "2026-10-20", "kind": "deck", "cat": "models", "tags": ["线性注意力", "架构", "长上下文"], "desc": "用可回放动画推导线性 attention 的 S/z 累加器、归一化读出、遗忘门与 delta 更新，理解固定状态和精确历史检索的取舍。"},
    {"slug": "paged-attention", "title": "PagedAttention：让 KV cache 按需生长", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["KV cache", "显存", "推理"], "desc": "用可回放动画理解 KV cache 分页、逻辑块到物理块的映射、按需分配与释放，以及共享前缀和写时复制。"},
    {"slug": "continuous-batching", "title": "Continuous batching：每一轮重新组队", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["调度", "推理", "KV cache"], "desc": "用动画看静态 batch 的空转、逐轮调度、新请求入队、decode 与 prefill 的区别，以及 chunked prefill 的 token 预算。"},
    {"slug": "prefix-caching", "title": "Prefix caching：相同的开头，只算一次", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["KV cache", "推理", "调度"], "desc": "用动画讲清 prefix cache 的精确前缀匹配、满块哈希链、KV 复用、共享引用计数、缓存命中与驱逐。"},
    {"slug": "speculative-decoding", "title": "投机解码", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["投机解码", "推理"], "desc": "用动画讲清并行验证、贪心前缀接受、p/q 接受概率、残差重采样、KV 回滚与投机解码的时间账本。"},
    {"slug": "dspark", "title": "DSpark", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["投机解码", "DeepSeek", "调度"], "desc": "用动画理解 DSpark 的并行 backbone、轻量 Markov head、条件接受率、前缀存活概率与负载感知验证调度。"},
    {"slug": "llm-quantization", "title": "LLM 量化", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["量化", "低精度", "显存"], "desc": "用动画讲 scale、round、clamp、反量化、离群值、per-group 精度、低位 GEMM 数据流，以及权重、激活与 KV cache 的独立账本。"},
    {"slug": "distributed-inference", "title": "分布式推理：数据怎样走过每张卡", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["推理", "并行", "通信", "MoE"], "desc": "用逐卡动画解释 HBM、权重、KV cache 与激活流动，连起 prefill、decode、TP、PP 和 EP。"},
    {"slug": "cpu-overhead", "title": "CPU overhead：GPU 为什么在等", "date": "2026-10-20", "kind": "deck", "cat": "inference", "tags": ["推理", "调度", "profiling"], "desc": "从 CPU 入队到 GPU 执行，用时间线动画讲异步提交、空闲间隔、同步、batching 与正确测量。"},
    {"slug": "cuda-graph", "title": "CUDA Graph：把一轮计算录下来", "date": "2026-10-20", "kind": "deck", "cat": "kernels", "tags": ["GPU", "kernel", "推理"], "desc": "动画解释 CUDA Graph 捕获、重放、固定地址、shape buckets、padding、内存池，以及与 kernel fusion 的区别。"},
    {"slug": "flash-attention", "title": "FlashAttention 1、2、3", "date": "2026-10-20", "kind": "deck", "cat": "kernels", "tags": ["GPU", "kernel", "注意力"], "desc": "从 attention 的显存往返讲起，用可拖动动画理解分块计算、online softmax、反向重算、FlashAttention-2 的 split-Q，以及 FlashAttention-3 在 Hopper 上的异步流水线。"},
    {"slug": "rmsnorm-kernel", "title": "RMSNorm 与 GPU 算子优化", "date": "2026-10-19", "kind": "deck", "cat": "kernels", "tags": ["GPU", "kernel", "RMSNorm"], "desc": "用动画讲 RMSNorm：公式、反向推导、FP32 累加，以及在 GPU 上把它写成一个跑满带宽的融合 kernel——向量化读、warp shuffle 归约、寄存器缓存、残差与 FP8 量化融合、Triton 实现和反向的两阶段 dg 归约。"},
    {"slug": "gradient-checkpointing", "title": "Gradient checkpointing", "date": "2026-10-19", "kind": "deck", "cat": "training", "tags": ["显存", "激活重算"], "desc": "用动画讲激活重算：激活为什么要存、显存锯齿、逐层 checkpoint 与 √n 策略、选择性重算、FlashAttention、CPU offload、与流水线并行和 FSDP 的配合，以及 torch.utils.checkpoint 的用法和一个可拖动的显存/算力权衡。"},
    {"slug": "optimizers", "title": "优化器：从 SGD 到 Muon", "date": "2026-10-19", "kind": "deck", "cat": "recipes", "tags": ["优化器"], "desc": "一套动画幻灯片：把 GD、动量、Nesterov、AdaGrad、RMSProp、Adam、AdamW、Lion、Adafactor、Shampoo 和 Muon 放到同一片 loss 地形上实际跑一遍，附可交互的优化器游乐场。"},

    { slug: "deepseek-v4", title: "DeepSeek-V4 技术报告精读", date: "2026-10-19", kind: "interactive", cat: "models", tags: ["DeepSeek", "架构", "稀疏注意力", "MoE", "长上下文", "优化器", "低精度"],
      desc: "拆解 DeepSeek-V4：CSA/HCA 注意力、mHC、Muon、FP4 QAT 与多教师蒸馏，附成本计算器。" },
    { slug: "qwen3-next", title: "Qwen3-Next：混合线性注意力与超稀疏 MoE", date: "2026-10-19", kind: "interactive", cat: "models", tags: ["Qwen", "架构", "线性注意力", "MoE", "投机解码"],
      desc: "从线性注意力推到 Gated DeltaNet、门控注意力、3:1 混合、512 专家 MoE 与 MTP。" },
    { slug: "kimi-k3", title: "Kimi K3：KDA、注意力残差与 Stable LatentMoE", date: "2026-10-19", kind: "interactive", cat: "models", tags: ["Kimi", "架构", "线性注意力", "MoE", "优化器", "长上下文"],
      desc: "精读 Kimi K3：有下界的 KDA、注意力残差、Stable LatentMoE 与分位数均衡，以及 1M 上下文系统。" },
    { slug: "kimi-k2", title: "Kimi K2：MuonClip 与万亿参数 Agent 模型", date: "2026-10-19", kind: "interactive", cat: "models", tags: ["Kimi", "架构", "优化器", "MoE", "RL"],
      desc: "精读 Kimi K2：Muon 与 QK-Clip 推导、数据改写、超稀疏 MoE、训练基础设施、Agent 数据合成与 RL。" },
    { slug: "kimi-linear", title: "Kimi Linear：Kimi Delta Attention", date: "2026-10-19", kind: "interactive", cat: "models", tags: ["Kimi", "线性注意力", "架构", "KV cache"],
      desc: "从线性注意力、DeltaNet 到 KDA：通道级门控、分块并行算法、3:1 混合与 NoPE、效率实验。" },
    { slug: "smol-decisions", title: "从目标到决策：训练什么、怎么做消融", date: "2026-10-19", kind: "interactive", cat: "recipes", tags: ["消融", "评测"],
      desc: "该不该训、训什么、怎样用小规模消融做决策，以及评测噪声与消融成本。" },
    { slug: "smol-architecture", title: "架构选择：注意力、位置编码、MoE 与混合模型", date: "2026-10-19", kind: "interactive", cat: "recipes", tags: ["架构", "注意力", "MoE", "KV cache"],
      desc: "MHA/GQA/MLA 与 KV cache、文档掩码、RoPE/NoPE、稳定性技巧、MoE 设计与混合模型。" },
    { slug: "smol-optim-data", title: "分词器、优化器、超参与数据配比", date: "2026-10-19", kind: "interactive", cat: "recipes", tags: ["优化器", "数据"],
      desc: "BPE 与分词器评估、AdamW 与 Muon、学习率调度、batch size 与超参 scaling law、数据配比。" },
    { slug: "smol-marathon", title: "训练马拉松：吞吐、loss spike 与中途调整", date: "2026-10-19", kind: "interactive", cat: "recipes", tags: ["稳定性", "长上下文"],
      desc: "SmolLM3 长跑实录：起跑清单、四个谜案、loss spike 诊断、多阶段 mid-training 与长上下文扩展。" },
    { slug: "smol-post-training", title: "后训练：SFT、偏好优化与 RL", date: "2026-10-19", kind: "interactive", cat: "recipes", tags: ["后训练", "RL"],
      desc: "后训练全流程：评测先行、chat template 与掩码、SFT 调参、推理 mid-training、DPO/APO 推导、GRPO 与长度惩罚。" },
    { slug: "smol-infrastructure", title: "基础设施：GPU 集群、通信与存储", date: "2026-10-19", kind: "interactive", cat: "training", tags: ["硬件", "通信", "GPU"],
      desc: "H100 内部、PCIe/NVLink/EFA 实测带宽、nccl-tests、存储分层、节点健康与 checkpoint，以及该用多少卡。" },
    { slug: "scaling-memory", title: "训练的账本：显存、FLOPs 与 MFU", date: "2026-10-19", kind: "interactive", cat: "training", tags: ["显存"],
      desc: "一步训练里显存花在哪、FLOPs 怎么数、MFU 怎么算：从 16 字节/参数推到 6ND，附显存计算器与训练时长估算器。" },
    { slug: "scaling-roofline", title: "Roofline：算力、带宽与算术强度", date: "2026-10-19", kind: "interactive", cat: "kernels", tags: ["硬件", "GPU"],
      desc: "一个运算到底卡在算力、显存带宽还是网络上：算术强度、roofline 图、矩阵乘法的临界 batch，以及 TPU 与 GPU 的硬件结构。" },
    { slug: "scaling-collectives", title: "集合通信与分片矩阵乘法", date: "2026-10-19", kind: "deck", cat: "training", tags: ["通信", "并行"],
      desc: "从 broadcast 到 all-to-all，推导环形集合通信的代价，再用分片记号讲清分片矩阵乘法。" },
    { slug: "scaling-data-parallel", title: "数据并行、ZeRO 与 FSDP", date: "2026-10-19", kind: "interactive", cat: "training", tags: ["并行", "显存", "通信"],
      desc: "从朴素数据并行讲到 ZeRO-1/2/3 与 FSDP：通信重叠、分桶、no_sync、显存公式与通信瓶颈。" },
    { slug: "scaling-tensor-parallel", title: "张量并行与序列并行", date: "2026-10-19", kind: "interactive", cat: "training", tags: ["并行", "通信"],
      desc: "从矩阵乘法的两种切法讲起：Megatron 张量并行、f/g 算子、词表并行、序列并行与通信量。" },
    { slug: "scaling-context-parallel", title: "上下文并行与 Ring Attention", date: "2026-10-19", kind: "interactive", cat: "training", tags: ["并行", "长上下文", "注意力"],
      desc: "长序列训练怎么切激活：online softmax 合并、Ring Attention、zig-zag 负载均衡与 Ulysses。" },
    { slug: "scaling-pipeline", title: "流水线并行：从 GPipe 到 DualPipe", date: "2026-10-19", kind: "deck", cat: "training", tags: ["并行"],
      desc: "交互幻灯片：推导 AFAB、1F1B、交错、Zero Bubble 与 DualPipe 的气泡与显存，附调度模拟器。" },
    { slug: "scaling-moe", title: "MoE 与专家并行", date: "2026-10-19", kind: "interactive", cat: "training", tags: ["MoE", "并行", "通信"],
      desc: "从路由器、容量、负载均衡损失到无辅助损失偏置，再到专家并行的 all-to-all 与通信成本。" },
    { slug: "scaling-5d", title: "5D 并行：组合与选型", date: "2026-10-19", kind: "interactive", cat: "training", tags: ["并行"],
      desc: "DP、TP、CP、PP、EP 各切什么、通信放在网络哪层、怎么组合，附配置规划器与设备网格演示。" },
    { slug: "scaling-jax", title: "JAX 分片编程与 Profiling", date: "2026-10-19", kind: "interactive", cat: "training", tags: ["JAX", "并行", "profiling"],
      desc: "用 mesh、PartitionSpec、shard_map 写分片程序，读懂 HLO 与 profile，分清计算、通信与空闲。" },
    { slug: "scaling-gpu-kernels", title: "GPU 内核与 FlashAttention", date: "2026-10-19", kind: "interactive", cat: "kernels", tags: ["GPU", "kernel", "注意力"],
      desc: "从 SM、warp、内存层级讲起，到访存合并、分块、融合、Triton，再到 FlashAttention 1–3 的推导与实现。" },
    { slug: "scaling-mixed-precision", title: "混合精度与 FP8 训练", date: "2026-10-19", kind: "interactive", cat: "kernels", tags: ["低精度", "量化"],
      desc: "浮点格式、舍入、损失缩放、FP32 主权重，到 FP8 的缩放策略、DeepSeek-V3 的细粒度量化与 MX/NVFP4 微缩放。" },
    { slug: "scaling-inference", title: "Transformer 推理的数学", date: "2026-10-19", kind: "interactive", cat: "inference", tags: ["推理", "KV cache", "投机解码", "并行"],
      desc: "从 KV cache 到投机解码：推导解码为什么受带宽限制、延迟下界、吞吐与延迟的权衡，以及 LLaMA-3-70B 的服务账本。" },
    { slug: "next-token", title: "一次只猜一个 token", date: "2026-10-19", kind: "deck", cat: "basics", tags: ["推理"],
      desc: "用 8 页动画讲清楚大模型生成文本的循环：切词、打分、采样、再来一遍。" },
    { slug: "hello-world", title: "新博客，从零开始", date: "2026-10-19", kind: "interactive", cat: "basics", tags: [],
      desc: "这个博客怎么搭的，以及一个可以拖动的 temperature 采样演示。" },
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
