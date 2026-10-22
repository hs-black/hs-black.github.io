/* Deterministic numerical scenes; all playback is reader controlled. */
(() => {
  'use strict';
  const get = id => document.getElementById(id);
  const sum = a => a.reduce((x, y) => x + y, 0);
  const dot = (a, b) => sum(a.map((v, i) => v * b[i]));
  const fixed = (x, n = 4) => (Math.abs(x) < 0.5 * 10 ** -n ? 0 : x).toFixed(n);
  const vector = a => '[' + a.map(v => fixed(v)).join(', ') + ']';
  function routing() {
    const z = [Number(get('router-z').value), 0, -1];
    const v = get('router-values').value === 'same' ? [2, 2, 2] : [2, 3, 1];
    const k = Number(get('router-k').value);
    const selected = get('router-mode').value === 'selected';
    const exps = z.map(x => Math.exp(x - Math.max(...z)));
    const p = exps.map(x => x / sum(exps));
    // z0 stays positive, so these scenes retain a fixed top-k order.
    const mass = sum(p.slice(0, k));
    const g = p.map((x, i) => i < k ? x / (selected ? mass : 1) : 0);
    const y = dot(g, v);
    const dy = p.map((x, j) => selected ? g[j] * (v[j] - y) : x * ((j < k ? v[j] : 0) - y));
    return { z, v, k, p, g, y, dy, selected };
  }
  const router = window.ArticleSteps.mount('#router-demo', {
    steps: [
      { title: '同一输入，三个专家', caption: '卡片列出各专家的输出分量 v 和 router 分数 z。先选择门控方式、Top-K 与专家是否已经分化。' },
      { title: '计算全体概率', caption: '三个分数经过同一次 softmax，得到和为 1 的概率 p。' },
      { title: '选择与归一化', caption: '边框标出本 token 调用的专家。条长和数值 g 都是实际参与加权的系数；未选中专家的 g 为 0。' },
      { title: '加权求和', caption: '各专家贡献 g × v 相加得到输出 y。先核对门控和，再与 dense 输出 2 比较。' },
      { title: '检查 router 导数', caption: '每列底部显示 ∂y/∂z。固定专家输出、轻微改变 z₀，可以观察当前选择分支的局部变化。' }
    ],
    render(_, stage) {
      const r = routing();
      get('router-z-out').textContent = r.z[0].toFixed(1);
      get('router-experts').innerHTML = r.v.map((v, i) => `
        <div class="up-expert${stage >= 2 && i < r.k ? ' selected' : ''}">
          <strong>专家 ${i}</strong><span>v = ${v}</span><span>z = ${fixed(r.z[i], 1)}</span>
          <span>p = ${stage >= 1 ? fixed(r.p[i]) : '—'}</span>
          <span>g = ${stage >= 2 ? fixed(r.g[i]) : '—'}</span>
          <span class="up-bar"><i style="width:${stage >= 2 ? 100 * r.g[i] : 0}%"></i></span>
          <span>g × v =<br>${stage >= 3 ? fixed(r.g[i] * v) : '—'}</span>
          <span>∂y/∂z =<br>${stage >= 4 ? fixed(r.dy[i]) : '—'}</span>
        </div>`).join('');
      get('router-sum').textContent = stage >= 2 ? fixed(sum(r.g)) : '等待门控';
      get('router-output').textContent = stage >= 3 ? fixed(r.y, 6) : '等待求和';
      const norm = Math.hypot(...r.dy);
      const texts = [
        `本次使用 top-${r.k}；${r.selected ? '选中集合内归一化' : '全体 softmax 后保留选中项'}。继续下一步查看三个概率。`,
        `全体概率之和为 ${fixed(sum(r.p))}。当前选中的前 ${r.k} 个专家共有 ${fixed(sum(r.p.slice(0, r.k)))} 的概率质量。`,
        `实际门控和为 ${fixed(sum(r.g))}。${r.selected ? '选中项除以选中概率总量，使门控和成为 1。' : '保留全体 softmax 的数值，未选中概率不参与输出。'}`,
        `y = ${r.g.map((g, i) => fixed(g) + ' × ' + r.v[i]).join(' + ')} = ${fixed(r.y, 6)}。相对 dense 值 2，差值为 ${fixed(r.y - 2, 6)}。`,
        `导数向量为 ${vector(r.dy)}，范数 ${fixed(norm, 6)}。${norm < 1e-12 ? '当前输出对这些 logits 的局部变化不敏感。改为分化后 top-2，或保留全体 softmax 权重，可观察新的梯度。' : '各位置的导数之和为 0，与同时平移所有 logits 保持概率不变相符。'}`
      ];
      get('router-readout').textContent = texts[stage];
    }
  });
  ['router-mode', 'router-k', 'router-values', 'router-z'].forEach(id => get(id).addEventListener('input', () => router.render()));

  const input = [1, -1];
  const wg = [[1, 0], [0, -1], [0.5, 0], [0.5, -0.5]];
  const wu = [[1, 0], [0, 1], [1, 0.5], [1, -1]];
  const wd = [[1, 0.5, -0.5, 1], [0.5, 1, 1, -0.5]];
  const hidden = wg.map((row, i) => {
    const u = dot(row, input);
    return u / (1 + Math.exp(-u)) * dot(wu[i], input);
  });
  const parts = [0, 1].map(part => wd.map(row => dot(row.slice(part * 2, part * 2 + 2), hidden.slice(part * 2, part * 2 + 2))));
  const dense = parts[0].map((v, i) => v + parts[1][i]);
  const shard = window.ArticleSteps.mount('#shard-demo', {
    steps: [
      { title: '计算四个隐藏通道', caption: 'h = SiLU(Wg x) ⊙ (Wu x)。每片卡片固定显示两个通道，数值由随文 lab 的同一组矩阵计算。' },
      { title: '沿通道切分', caption: '将 gate/up 的对应行和 down 的对应列配对。每片得到二维输出贡献 A 或 B；相加就是 dense 输出。' },
      { title: '复制每个分片', caption: 'A₀、A₁ 共用 A 的初始参数，B₀、B₁ 共用 B 的初始参数。四个专家各包含两个隐藏通道。' },
      { title: '选出两个专家', caption: '实色标签表示本 token 选中的副本。先检查是否同时覆盖 A 和 B，再核对门控值。' },
      { title: '缩放、门控与求和', caption: '先把专家输出乘 down 权重倍数，再乘门控，最后求和。输出误差使用二维向量的 L2 范数。' }
    ],
    render(_, stage) {
      const complete = get('shard-choice').value === 'complete';
      const gates = Number(get('shard-gate').value);
      const scale = Number(get('shard-scale').value);
      const picks = complete ? [1, 1] : [2, 0];
      const output = dense.map((_, j) => gates * scale * sum(parts.map((part, i) => picks[i] * part[j])));
      const error = Math.hypot(...dense.map((v, j) => v - output[j]));
      get('shard-cards').innerHTML = parts.map((part, i) => `
        <div class="up-shard">
          <strong>分片 ${i ? 'B' : 'A'} · 通道 ${2 * i}、${2 * i + 1}</strong>
          <code>h = ${vector(hidden.slice(i * 2, i * 2 + 2))}</code>
          <code>输出贡献 =<br>${stage >= 1 ? vector(part) : '等待 down 投影'}</code>
          <div class="up-replicas${stage < 2 ? ' up-muted' : ''}">
            ${[0, 1].map(copy => `<span class="${stage >= 3 && copy < picks[i] ? 'picked' : ''}">${i ? 'B' : 'A'}${copy ? '₁' : '₀'}</span>`).join('')}
          </div>
          <code>${stage >= 3 ? '选中 ' + picks[i] + ' 份；各 g = ' + gates : '等待选择'}</code>
        </div>`).join('');
      get('shard-output').textContent = stage >= 4 ? vector(output) : '等待组合';
      get('shard-error').textContent = stage >= 4 ? 'L2 误差 = ' + fixed(error, 6) : 'L2 误差 = —';
      const texts = [
        `四个通道的值为 ${vector(hidden)}。A 使用前两个通道，B 使用后两个通道。`,
        `A = ${vector(parts[0])}，B = ${vector(parts[1])}；A + B = ${vector(dense)}。`,
        '两个分片各复制两次，专家总数变为 4；每次只调用其中 2 个。副本的初始输出与所属分片相同。',
        `本次选择 ${complete ? 'A₀ + B₀，覆盖两片' : 'A₀ + A₁，两次使用 A 片'}。每个门控 g = ${gates}，down 倍数 = ${scale}。`,
        `输出 = ${gates} × ${scale} × ${complete ? '(A + B)' : '(A + A)'} = ${vector(output)}。${error < 1e-10 ? '分片覆盖与输出幅度都对齐，精确恢复 dense 输出。' : complete ? '已覆盖两片；将门控与 down 倍数的乘积调为 1，即可恢复幅度。' : 'B 的输出方向没有进入组合。先选齐 A、B，再校正整体幅度。'}`
      ];
      get('shard-readout').textContent = texts[stage];
    }
  });
  ['shard-choice', 'shard-gate', 'shard-scale'].forEach(id => get(id).addEventListener('input', () => shard.render()));
})();
