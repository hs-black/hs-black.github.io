/* IQuest-Q1: exact cache accounting, sink probability and halo merge. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const number = value => Number(value).toLocaleString('en-US');
  const figures = {};
  const GiB = 2 ** 30;
  const lengths = [4096, 32768, 131072, 524288];
  function cacheAt(tokens) {
    const global = 25 * tokens * 4096 / GiB;
    const local = 63 * Math.min(tokens, 4096) * 4096 / GiB;
    const reference = 88 * tokens * 4096 / GiB;
    return {tokens, global, local, total: global + local, reference};
  }
  if ($('iq-cache')) {
    const map = $('iq-layer-map');
    for (let i = 0; i < 88; i++) {
      const full = i < 2 || i >= 86 || (i - 2) % 4 === 3;
      const cell = document.createElement('span');
      cell.className = 'iq-layer ' + (full ? 'iq-global' : 'iq-local');
      cell.title = `第 ${i+1} 层：${full ? '全局注意力' : '滑动窗口注意力'}`;
      cell.innerHTML = `<span>${i + 1}</span><b>${full ? 'G' : 'S'}</b>`;
      map.append(cell);
    }
    figures.cache = ArticleSteps.mount('#iq-cache', {
      duration: 1800,
      steps: [
        {title:'窗口刚好填满', caption:'4,096 个位置都在窗口内。两种注意力此时都保存完整历史。'},
        {title:'增长到 32K', caption:'全局层保存 32,768 个位置。窗口层为新位置腾出空间，仍各自保存最近 4,096 个位置。'},
        {title:'增长到 128K', caption:'窗口层的 KV 合计保持约 0.984 GiB；新增容量主要来自 25 个全局层。'},
        {title:'增长到 512K', caption:'混合 KV 合计 50.984 GiB；同样头数与精度的 88 层全局对照为 176 GiB。'}
      ],
      render(step, index, progress) {
        const tokens = Math.round(lengths[index] + (lengths[Math.min(index+1,3)] - lengths[index]) * progress);
        const m = cacheAt(tokens);
        $('iq-cache-length').value = String(index);
        $('iq-context-tokens').textContent = number(tokens);
        $('iq-global-size').textContent = m.global.toFixed(3) + ' GiB';
        $('iq-local-size').textContent = m.local.toFixed(3) + ' GiB';
        $('iq-reference-size').textContent = m.reference.toFixed(3) + ' GiB';
        $('iq-total-size').textContent = m.total.toFixed(3);
        $('iq-cache-ratio').textContent = (100 * m.total/m.reference).toFixed(1) + '%';
        $('iq-global-bar').style.width = `${100 * m.global/176}%`;
        $('iq-local-bar').style.width = `${100 * m.local/176}%`;
        $('iq-reference-bar').style.width = `${100 * m.reference/176}%`;
        $('iq-cache').dataset.totalGib = m.total;
        $('iq-cache').dataset.referenceGib = m.reference;
      }
    });
    $('iq-cache-length').addEventListener('change', e => figures.cache.seek(Number(e.target.value)));
  }
  function sinkAt(logit, activation = 1) {
    const weight = Math.exp(logit) * activation;
    const denominator = 4 + weight;
    return {weight, denominator, probabilities: [2, 1, 1, weight].map(x => x / denominator), output: 4 / denominator};
  }
  if ($('iq-sink')) {
    figures.sink = ArticleSteps.mount('#iq-sink', {
      duration: 1700,
      steps: [
        {title:'先看三个历史位置', caption:'exp(logit) 为 2、1、1，分母为 4。概率为 1/2、1/4、1/4，加权输出为 1。'},
        {title:'加入 sink 的分数', caption:'sink 增加一个正的 exp(logit)，四个位置共享更大的分母。默认分数使分母从 4 变成 8。'},
        {title:'一起归一化', caption:'四个位置共享同一个分母。sink 获得概率，三个历史位置维持 2:1:1 的相对比例。'},
        {title:'读出加权结果', caption:'sink 的 V 为 0，因此新增的概率只调小历史输出。改变 sink logit，可以观察这个缩放随分数变化。'}
      ],
      render(step, index, progress) {
        const logit = Number($('iq-sink-score').value);
        const activation = index === 0 ? progress : 1;
        const m = sinkAt(logit, activation);
        const values = [1, 3, -1, 0];
        $('iq-sink-score-value').textContent = logit.toFixed(3);
        m.probabilities.forEach((p, i) => {
          document.querySelector(`[data-sink-bar="${i}"]`).style.width = `${100*p}%`;
          document.querySelector(`[data-sink-prob="${i}"]`).textContent = `${(100*p).toFixed(1)}%`;
          document.querySelector(`[data-sink-term="${i}"]`).textContent = `${p.toFixed(3)} × ${values[i] < 0 ? '−1' : values[i]}`;
        });
        $('iq-sink-denominator').textContent = m.denominator.toFixed(3);
        $('iq-sink-weight').textContent = m.weight.toFixed(3);
        $('iq-sink-history').textContent = `${(100 * (1 - m.probabilities[3])).toFixed(1)}%`;
        $('iq-sink-output').textContent = m.output.toFixed(3);
        $('iq-sink-equation').textContent = `(2 × 1 + 1 × 3 + 1 × −1 + ${m.weight.toFixed(3)} × 0) / ${m.denominator.toFixed(3)} = ${m.output.toFixed(3)}`;
        $('iq-sink').dataset.output = m.output;
        $('iq-sink').dataset.mass = m.probabilities.reduce((a,b) => a+b,0);
        $('iq-sink').dataset.stage = index;
      }
    });
    $('iq-sink-score').addEventListener('input', () => figures.sink.render());
  }
  if ($('iq-halo')) {
    const status = [
      'halo 正在传输；本卡可以先算自己的部分。',
      'halo 到达，得到邻卡部分的输出 1 和 LSE = ln 2。',
      '合并分母：4 + 2 = 6。本卡权重 2/3，邻卡权重 1/3。',
      '加权后得到 7/3，与把三个位置放在一起计算的结果相同。'
    ];
    figures.halo = ArticleSteps.mount('#iq-halo', {
      duration: 1600,
      steps: [
        {title:'本地先算', caption:'本卡两项 exp(logit) 相加为 4，加权值之和为 12，因此局部输出为 3。'},
        {title:'接到远端 halo', caption:'邻卡这一项 exp(logit) 为 2，V 为 1，因此局部输出为 1。'},
        {title:'恢复全局的归一化比例', caption:'将两个分区的 Z 相加，再用 Z_local / Z_total 和 Z_halo / Z_total 加权。'},
        {title:'得到完整窗口输出', caption:'2/3 × 3 + 1/3 × 1 = 7/3。记录 LSE，便能在数值稳定的形式下恢复这两个比例。'}
      ],
      render(step, index, progress) {
        $('iq-halo-status').textContent = status[index];
        const remoteOpacity = index === 0 ? 0.4 + 0.6 * progress : 1;
        $('iq-halo-remote').style.opacity = remoteOpacity;
        $('iq-halo-remote').classList.toggle('iq-halo-pending', index === 0);
        $('iq-halo-merge').style.opacity = index < 1 ? 0 : index === 1 ? progress : 1;
        $('iq-halo').dataset.output = 7 / 3;
      }
    });
  }
  window.IQuestFigures = {figures, cacheAt, sinkAt};
})();
