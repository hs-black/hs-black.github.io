(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const fmt = x => Number(x).toFixed(8).replace(/0+$/, '').replace(/\.$/, '');
  const vector = values => '[' + values.map(fmt).join(', ') + ']';
  const taps = [.1, .2, .3, .4];
  function consume(state, value) {
    const window = [...state.history, value];
    const v = window.reduce((sum, x, i) => sum + taps[i] * x, 0);
    return {s: .5 * state.s + .5 * v, history: window.slice(1), kv: [...state.kv, value], v, window};
  }
  function replay(values) {
    return values.reduce(consume, {s: 0, history: [0, 0, 0], kv: [], v: 0, window: [0, 0, 0, 0]});
  }
  function statePanel(name, state, accent) {
    return '<section class="qne-panel"><h4>' + name + '</h4><dl>' +
      '<dt>GDN 状态 S</dt><dd class="' + (accent ? 'qne-emphasis' : '') + '">' + fmt(state.s) + '</dd>' +
      '<dt>卷积历史：前 3 个 u</dt><dd>' + vector(state.history) + '</dd>' +
      '<dt>KV 条目：每项 (K, V) = (u, u)</dt><dd>' + state.kv.map(v => '(' + fmt(v) + ',' + fmt(v) + ')').join(' ') + '</dd>' +
      '<dt>已消费位置数</dt><dd>' + state.kv.length + '</dd>' +
      '<dt>最近写入：卷积窗口 → v</dt><dd>' + vector(state.window) + ' → ' + fmt(state.v) + '</dd>' +
      '</dl></section>';
  }
  const prefix = replay([1, 2, 3, 4]);
  const accepted = consume(prefix, 5);
  const proposed = consume(accepted, 6);
  function renderState(_, step) {
    const replacement = +$('hybrid-correction').value;
    $('hybrid-correction-value').textContent = replacement;
    const kvOnly = {...proposed, kv: accepted.kv.slice()};
    const nextCorrect = consume(accepted, replacement), nextWrong = consume(kvOnly, replacement);
    let correct = prefix, wrong = prefix;
    if (step === 1) correct = wrong = accepted;
    if (step === 2 || step === 3) correct = wrong = proposed;
    if (step === 4) {correct = accepted; wrong = kvOnly;}
    if (step === 5) {correct = nextCorrect; wrong = nextWrong;}
    $('hybrid-state-panels').innerHTML = statePanel('恢复 S、卷积历史与 KV', correct, step >= 4) + statePanel('仅把 KV 截到接受前缀', wrong, false);
    const tokens = [1, 2, 3, 4];
    if (step >= 1) tokens.push(5);
    if (step >= 2) tokens.push(6);
    if (step === 5) tokens.push(replacement);
    $('hybrid-token-chips').innerHTML = tokens.map((v, i) => '<span class="qne-chip ' + (i === 5 && step >= 3 ? 'reject' : i >= 4 && step < 3 ? 'pending' : i === 4 || i === 6 ? 'accept' : '') + '">' + v + '</span>').join('');
    const messages = [
      '已消费的四个投影为 [1, 2, 3, 4]。当前 S = 2.1625，卷积历史为 [2, 3, 4]，KV 有 4 项。',
      '卷积得到 v = 4，状态更新为 S = 0.5 × 2.1625 + 0.5 × 4 = 3.08125。',
      '卷积得到 v = 5，试探状态更新为 S = 4.040625。第 6 个位置已经影响 S 和卷积历史。',
      '正确的恢复边界是值 5 之后的状态：S = 3.08125、卷积历史 [3, 4, 5]、KV 前 5 项。',
      '完整恢复后，三类状态都对应接受前缀。仅截 KV 的分支仍保留第 6 个位置的 S 与卷积历史。',
      '完整恢复：v = ' + fmt(nextCorrect.v) + '，读出 S = ' + fmt(nextCorrect.s) + '\n仅截 KV：v = ' + fmt(nextWrong.v) + '，读出 S = ' + fmt(nextWrong.s) + '\n两条读出的差 = ' + fmt(nextWrong.s - nextCorrect.s) + '。'
    ];
    $('hybrid-state-result').textContent = messages[step];
    const root = $('hybrid-state-demo');
    root.dataset.correctS = nextCorrect.s; root.dataset.wrongS = nextWrong.s;
    root.dataset.correctV = nextCorrect.v; root.dataset.wrongV = nextWrong.v;
    root.dataset.step = step;
  }
  const state = ArticleSteps.mount('#hybrid-state-demo', {
    duration: 1800,
    steps: [
      {title: '记录前缀状态', caption: '两条路径从同一组状态开始。'},
      {title: '执行第一枚候选', caption: '消费投影值 5，更新卷积、GDN 与 KV。'},
      {title: '执行第二枚候选', caption: '消费投影值 6，暂时把两枚候选都计入状态。'},
      {title: '得到验证结果', caption: '第一枚被接受，第二枚被拒绝。划线的格子表示被拒绝的候选。'},
      {title: '恢复到同一位置', caption: '左侧恢复完整检查点，右侧只裁掉 KV 尾项。'},
      {title: '消费修正 token', caption: '从接受前缀继续执行，对照新的卷积结果与 GDN 读出。'}
    ], render: renderState
  });
  $('hybrid-correction').addEventListener('input', () => {state.pause(); state.render();});

  function generator(seed) {
    return () => {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  const routeCache = new Map();
  function sampleRoutes(batch) {
    if (routeCache.has(batch)) return routeCache.get(batch);
    const random = generator(20261022), counts = Array(512).fill(0), routes = [];
    for (let token = 0; token < batch; token++) {
      const experts = new Set();
      while (experts.size < 10) experts.add(Math.floor(random() * 512));
      const ids = [...experts].sort((a, b) => a - b);
      for (const e of ids) counts[e]++;
      if (token < 2) routes.push(ids);
    }
    const active = counts.filter(x => x > 0).length;
    const padded = counts.reduce((sum, m) => sum + 16 * Math.ceil(m / 16), 0);
    const result = {counts, routes, active, padded, assignments: 10 * batch, expected: 512 * (1 - (1 - 10 / 512) ** batch)};
    routeCache.set(batch, result); return result;
  }
  function stat(label, value) {return '<div class="qne-stat"><span>' + label + '</span><b>' + value + '</b></div>';}
  function histogram(counts, visible) {
    const maximum = Math.max(...counts), width = Math.max(1, Math.ceil(maximum / 8));
    const bins = [{lo: 0, hi: 0, count: counts.filter(m => m === 0).length}];
    for (let lo = 1; lo <= maximum; lo += width) bins.push({lo, hi: lo + width - 1, count: counts.filter(m => m >= lo && m < lo + width).length});
    return '<h4>每位专家收到多少 token 行 M</h4>' + bins.map(b => '<div class="qne-hist-row ' + (b.lo === 0 ? 'zero' : '') + '"><span>' + (b.lo === b.hi ? b.lo : b.lo + '–' + b.hi) + ' 行</span><div class="qne-hist-bar"><i style="width:' + (visible ? b.count / 512 * 100 : 0) + '%"></i></div><span>' + (visible ? b.count : '…') + ' 位</span></div>').join('');
  }
  function renderExperts(_, step) {
    const batch = +$('expert-batch-size').value, data = sampleRoutes(batch);
    const average = data.assignments / data.active, efficiency = 100 * data.assignments / data.padded;
    $('expert-routes').innerHTML = data.routes.map((ids, i) => '<div class="qne-route">token ' + i + ' → ' + ids.join(', ') + '</div>').join('') + (batch > 2 ? '<p class="qne-muted">其余 ' + (batch - 2) + ' 个 token 按同一规则独立选择。</p>' : '');
    $('expert-histogram').innerHTML = histogram(data.counts, step >= 1);
    $('expert-stats').innerHTML = stat('token→expert 分配总数', data.assignments.toLocaleString('en-US')) + stat('实际活跃专家 / 512', step >= 1 ? data.active : '待汇总') + stat('活跃专家平均 M', step >= 1 ? average.toFixed(2) : '待汇总') + stat('16 行 tile 的有效行比例', step >= 2 ? efficiency.toFixed(2) + '%' : '待填充');
    const maxM = Math.max(...data.counts);
    const e = data.counts.reduce((best, m, i) => m > 0 && (best < 0 || Math.abs(m - average) < Math.abs(data.counts[best] - average)) ? i : best, -1);
    const m = data.counts[e], rows = Math.ceil(m / 16) * 16;
    $('expert-tiles').innerHTML = '<h4>' + (step >= 2 ? '专家 ' + e + '：' + m + ' 个有效行 → ' + rows + ' 个执行行' : '下一步查看一位专家的 tile 填充') + '</h4><div class="qne-tile-grid">' + Array.from({length: rows}, (_, i) => '<span class="qne-tile-cell ' + (step >= 2 && i < m ? 'active' : '') + '" title="' + (i < m ? '有效 token 行' : '填充行') + '"></span>').join('') + '</div>';
    $('expert-batch-result').textContent = step === 0 ? '每个 token 选中 10 位不同专家；expert 编号从 0 到 511。所有 batch 大小复用同一固定种子的路由前缀。' : step === 1 ? '独立均匀选择下，活跃专家期望为 512 × [1 − (1 − 10/512)^B] = ' + data.expected.toFixed(2) + '。本次实际活跃 ' + data.active + ' 位，最大 M = ' + maxM + '。' : '全体专家共执行 ' + data.padded.toLocaleString('en-US') + ' 行，其中 ' + data.assignments.toLocaleString('en-US') + ' 行有效。有效行比例 = ' + data.assignments + ' / ' + data.padded + ' = ' + efficiency.toFixed(2) + '%。';
    const root = $('expert-batch-demo');
    root.dataset.active = data.active; root.dataset.expected = data.expected; root.dataset.padded = data.padded;
    root.dataset.efficiency = efficiency; root.dataset.assignments = data.assignments; root.dataset.counts = JSON.stringify(data.counts); root.dataset.step = step;
  }
  const experts = ArticleSteps.mount('#expert-batch-demo', {
    duration: 1800,
    steps: [
      {title: '每个 token 选择 10 位专家', caption: '展示前两个 token 的具体路由，所有编号来自固定种子的均匀采样。'},
      {title: '按专家聚合 token', caption: '横条长度表示该 M 区间内的专家数量；灰条表示未收到 token 的专家。'},
      {title: '按 16 行 tile 对齐', caption: '示例选择 M 最接近活跃平均值的专家。每格是一行，每 16 格为一个 tile；紫格有效，空格填充。'}
    ], render: renderExperts
  });
  $('expert-batch-size').addEventListener('change', () => {experts.pause(); experts.render();});
})();
