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
        ['scaling-mixed-precision', '混合精度与 FP8 训练'],
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
    reports: {
      title: '模型技术报告精读',
      posts: [
        ['deepseek-v4', 'DeepSeek-V4 技术报告精读'],
        ['qwen3-next', 'Qwen3-Next：混合线性注意力与超稀疏 MoE'],
        ['kimi-k2', 'Kimi K2：MuonClip 与万亿参数 Agent 模型'],
        ['kimi-linear', 'Kimi Linear：Kimi Delta Attention'],
      ],
    },
  };

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
