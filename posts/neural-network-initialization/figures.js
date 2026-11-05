/* Initialization figures: exact recurrences and a balanced scalar gradient flow.
 * All figures start paused. Shared ArticleSteps owns playback and visibility.
 */
(() => {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const number = (x, digits = 4) => Math.abs(x) > 0 && (Math.abs(x) < .0001 || Math.abs(x) >= 10000) ? x.toExponential(2) : x.toFixed(digits);
  function node(tag, attrs = {}, content) {
    const n = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([key, value]) => n.setAttribute(key, value));
    if (content !== undefined) n.textContent = content;
    return n;
  }
  function add(parent, tag, attrs, content) { const n = node(tag, attrs, content); parent.append(n); return n; }
  function stat(root, name, value) { root.querySelector('[data-stat="' + name + '"]').textContent = value; }
  function value(root, name, value) { root.querySelector('[data-value="' + name + '"]').textContent = value; }
  function chart(host, {xMax, yMin, yMax, xTicks, yTicks, xLabel, yLabel = '', yFormat = x => String(x)}) {
    const width = Math.max(250, Math.round(host.getBoundingClientRect().width));
    const height = Math.round(host.getBoundingClientRect().height) || 260;
    const margin = {left: 44, right: 14, top: 24, bottom: 38};
    const x = t => margin.left + t / xMax * (width - margin.left - margin.right);
    const y = t => height - margin.bottom - (t - yMin) / (yMax - yMin) * (height - margin.top - margin.bottom);
    const svg = node('svg', {viewBox: `0 0 ${width} ${height}`, 'aria-hidden': 'true'});
    host.replaceChildren(svg);
    yTicks.forEach(t => {
      add(svg, 'line', {x1: x(0), x2: x(xMax), y1: y(t), y2: y(t), class: 'init-grid'});
      add(svg, 'text', {x: margin.left - 7, y: y(t) + 4, 'text-anchor': 'end'}, yFormat(t));
    });
    xTicks.forEach(t => add(svg, 'text', {x: x(t), y: height - margin.bottom + 18, 'text-anchor': 'middle'}, String(t)));
    add(svg, 'line', {x1: x(0), x2: x(xMax), y1: y(yMin), y2: y(yMin), class: 'init-axis'});
    add(svg, 'text', {x: x(0), y: 13}, yLabel);
    add(svg, 'text', {x: x(xMax), y: height - 2, 'text-anchor': 'end'}, xLabel);
    return {svg, x, y, xMax, yMin, yMax};
  }
  function path(c, points, className) {
    if (!points.length) return;
    add(c.svg, 'path', {d: points.map(([x, y], i) => `${i ? 'L' : 'M'}${c.x(x).toFixed(3)},${c.y(y).toFixed(3)}`).join(' '), class: className});
  }
  function cursor(c, position, current) {
    add(c.svg, 'line', {x1: c.x(position), x2: c.x(position), y1: c.y(c.yMin), y2: c.y(c.yMax), class: 'init-cursor'});
    add(c.svg, 'circle', {cx: c.x(position), cy: c.y(current), r: 4.2, class: 'init-dot'});
  }
  function reference(c, y) { add(c.svg, 'line', {x1: c.x(0), x2: c.x(c.xMax), y1: c.y(y), y2: c.y(y), class: 'init-reference'}); }
  function followResize(root, render) {
    let lastWidth = -1;
    const observer = new ResizeObserver(entries => {
      const width = entries[0].contentRect.width;
      if (Math.abs(width - lastWidth) > 1) { lastWidth = width; render(); }
    });
    observer.observe(root);
  }
  function mountScale() {
    const root = document.getElementById('init-scale-figure');
    if (!root) return;
    const control = root.querySelector('[data-control="variance"]');
    let position = 0;
    function render() {
      const variance = Number(control.value);
      value(root, 'variance', variance.toFixed(2));
      const logMultiplier = Math.log10(variance / 2);
      const c = chart(root.querySelector('[data-chart="scale"]'), {xMax: 24, yMin: -10, yMax: 3, xTicks: [0, 6, 12, 18, 24], yTicks: [-9, -6, -3, 0, 3], xLabel: '层数 ℓ', yLabel: '预激活 qℓ · 对数轴', yFormat: t => ({'-9': '10⁻⁹', '-6': '10⁻⁶', '-3': '10⁻³', '0': '1', '3': '10³'})[t]});
      reference(c, 0);
      path(c, [[0, 0], [24, 24 * logMultiplier]], 'init-future');
      path(c, [[0, 0], [24, 24 * Math.log10(.5)]], 'init-future');
      path(c, [[0, 0], [position, position * Math.log10(.5)]], 'init-alt');
      path(c, [[0, 0], [position, position * logMultiplier]], 'init-path');
      cursor(c, position, position * logMultiplier);
      const depth = Math.floor(position + 1e-8);
      const q = Math.pow(variance / 2, depth);
      stat(root, 'q', number(q)); stat(root, 'r', number(q / 2));
      stat(root, 'var', number(q * (.5 - 1 / (2 * Math.PI)))); stat(root, 'mean', number(Math.sqrt(q / (2 * Math.PI))));
      root.querySelector('[data-chart="scale"]').setAttribute('aria-label', `第 ${depth} 层，预激活二阶矩 ${number(q)}，ReLU 二阶矩 ${number(q / 2)}`);
    }
    const api = ArticleSteps.mount(root, {duration: 330, steps: Array.from({length: 25}, (_, i) => ({title: `第 ${i} 层`, caption: i === 0 ? 'q₀ = 1。先经过 ReLU，二阶矩减半；再乘随机权重，二阶矩乘以 σw²。' : '本层 qℓ = (σw² / 2) × 上一层 qℓ₋₁。下方数字给出本层 ReLU 输出的二阶矩、方差与均值。'})), render(step, index, progress) { position = index + progress; render(); }});
    control.addEventListener('input', () => { api.pause(); render(); });
    root.querySelectorAll('[data-preset]').forEach(button => button.addEventListener('click', () => { control.value = button.dataset.preset; api.pause(); render(); }));
    followResize(root, render);
  }
  function correlationStep(c) { return (Math.sqrt(Math.max(0, 1 - c * c)) + (Math.PI - Math.acos(clamp(c, -1, 1))) * c) / Math.PI; }
  function mountCorrelation() {
    const root = document.getElementById('init-correlation-figure');
    if (!root) return;
    const control = root.querySelector('[data-control="correlation"]');
    let position = 0;
    function render() {
      const start = Number(control.value), cs = [start];
      for (let i = 0; i < 80; i++) cs.push(correlationStep(cs[i]));
      value(root, 'correlation', start.toFixed(2));
      const depth = Math.floor(position + 1e-8), progress = position - depth;
      const current = cs[depth] + (depth < 80 ? progress * (cs[depth + 1] - cs[depth]) : 0);
      [['correlation', x => x, -1, 1, [-1, 0, 1]], ['distance', x => Math.sqrt(Math.max(0, 2 - 2 * x)), 0, 2, [0, 1, 2]]].forEach(([name, transform, yMin, yMax, yTicks]) => {
        const c = chart(root.querySelector('[data-chart="' + name + '"]'), {xMax: 80, yMin, yMax, xTicks: [0, 40, 80], yTicks, xLabel: '层数 ℓ'});
        const points = cs.map((v, i) => [i, transform(v)]);
        path(c, points, 'init-future');
        path(c, points.slice(0, depth + 1).concat([[position, transform(current)]]), 'init-path');
        cursor(c, position, transform(current));
      });
      stat(root, 'correlation', number(cs[depth])); stat(root, 'distance', number(Math.sqrt(Math.max(0, 2 - 2 * cs[depth]))));
    }
    const api = ArticleSteps.mount(root, {duration: 150, steps: Array.from({length: 81}, (_, i) => ({title: `第 ${i} 层`, caption: i === 0 ? '两条预激活的二阶矩都为 1。c₀ 决定它们最初有多相似。' : '当前层先做 ReLU，再经过同一组随机权重。相关性继续向 1 靠近，单位二阶矩下的距离随之缩小。'})), render(step, index, progress) { position = index + progress; render(); }});
    control.addEventListener('input', () => { api.pause(); render(); });
    followResize(root, render);
  }
  function circlePlot(host, s1, s2, angle, output) {
    const width = Math.max(250, Math.round(host.getBoundingClientRect().width));
    const height = 215, cx = width / 2, cy = height / 2, unit = Math.min((width - 64) / 3.2, (height - 36) / 3.2);
    const svg = node('svg', {viewBox: `0 0 ${width} ${height}`, 'aria-hidden': 'true'});
    host.replaceChildren(svg);
    add(svg, 'line', {x1: cx - 1.6 * unit, x2: cx + 1.6 * unit, y1: cy, y2: cy, class: 'init-grid'});
    add(svg, 'line', {x1: cx, x2: cx, y1: cy - 1.6 * unit, y2: cy + 1.6 * unit, class: 'init-grid'});
    add(svg, 'circle', {cx, cy, r: unit, class: 'init-reference'});
    if (output) add(svg, 'ellipse', {cx, cy, rx: unit * s1, ry: unit * s2, class: 'init-alt'});
    [-1, 1].forEach(t => { add(svg, 'text', {x: cx + t * unit, y: cy + 17, 'text-anchor': 'middle'}, String(t)); add(svg, 'text', {x: cx - 8, y: cy - t * unit + 4, 'text-anchor': 'end'}, String(t)); });
    const tipX = cx + unit * s1 * Math.cos(angle), tipY = cy - unit * s2 * Math.sin(angle);
    add(svg, 'line', {x1: cx, y1: cy, x2: tipX, y2: tipY, class: output ? 'init-alt' : 'init-path', 'stroke-width': 3});
    const direction = Math.atan2(tipY - cy, tipX - cx), length = Math.hypot(tipX - cx, tipY - cy), arrow = Math.min(8, length * .5);
    const points = [[tipX, tipY], [tipX - arrow * Math.cos(direction - .45), tipY - arrow * Math.sin(direction - .45)], [tipX - arrow * Math.cos(direction + .45), tipY - arrow * Math.sin(direction + .45)]];
    add(svg, 'polygon', {points: points.map(p => p.join(',')).join(' '), fill: output ? 'var(--accent-2)' : 'var(--accent)'});
    add(svg, 'circle', {cx, cy, r: 2.2, fill: 'var(--text-faint)'});
  }
  function mountSpectrum() {
    const root = document.getElementById('init-spectrum-figure');
    if (!root) return;
    const control = root.querySelector('[data-control="minor"]');
    let position = 0;
    function render() {
      const minor = Number(control.value), major = Math.sqrt(2 - minor * minor), angle = position * 5 * Math.PI / 180;
      value(root, 'minor', minor.toFixed(2));
      circlePlot(root.querySelector('[data-chart="input-circle"]'), 1, 1, angle, false);
      circlePlot(root.querySelector('[data-chart="output-ellipse"]'), major, minor, angle, true);
      stat(root, 'singular', `${major.toFixed(4)}，${minor.toFixed(4)}`);
      stat(root, 'average', ((major * major + minor * minor) / 2).toFixed(4));
      stat(root, 'gain', Math.hypot(major * Math.cos(angle), minor * Math.sin(angle)).toFixed(4));
      stat(root, 'condition', (major / minor).toFixed(2));
    }
    const api = ArticleSteps.mount(root, {duration: 400, steps: Array.from({length: 19}, (_, i) => ({title: `输入方向 ${i * 5}°`, caption: i === 0 ? '沿横轴的扰动受到较大奇异值的缩放。改变 ε，可以让椭圆逐渐恢复成圆。' : i === 18 ? '沿纵轴的扰动只剩 ε 倍。即使平均平方奇异值为 1，这个方向也可能被明显压缩。' : '输入箭头始终长 1；输出箭头的长度由该方向在两个奇异向量上的分量共同决定。'})), render(step, index, progress) { position = index + progress; render(); }});
    control.addEventListener('input', () => { api.pause(); render(); });
    root.querySelectorAll('[data-preset]').forEach(button => button.addEventListener('click', () => { control.value = button.dataset.preset; api.pause(); render(); }));
    followResize(root, render);
  }
  function flow(depth, a0) {
    const rate = a => depth * Math.pow(Math.max(0, a), 2 - 2 / depth) * (1 - a);
    const dt = .02, points = [a0];
    let a = a0;
    for (let step = 1; step <= 6000; step++) {
      const k1 = rate(a), k2 = rate(a + dt * k1 / 2), k3 = rate(a + dt * k2 / 2), k4 = rate(a + dt * k3);
      a += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6;
      if (step % 50 === 0) points.push(a);
    }
    return points;
  }
  function mountDynamics() {
    const root = document.getElementById('init-dynamics-figure');
    if (!root) return;
    const product = root.querySelector('[data-control="product"]'), depthControl = root.querySelector('[data-control="depth"]');
    let position = 0, cacheKey = '', series, baseline;
    function render() {
      const a0 = Math.pow(10, Number(product.value)), depth = Number(depthControl.value), key = `${a0}:${depth}`;
      if (key !== cacheKey) { series = flow(depth, a0); baseline = flow(2, a0); cacheKey = key; }
      value(root, 'product', number(a0));
      const c = chart(root.querySelector('[data-chart="dynamics"]'), {xMax: 120, yMin: 0, yMax: 1.07, xTicks: [0, 30, 60, 90, 120], yTicks: [0, .5, 1], xLabel: '梯度流时间 t', yLabel: '输出模态强度 a(t)'});
      reference(c, 1);
      const time = Math.floor(position + 1e-8), progress = position - time;
      [baseline, series].forEach((values, i) => {
        const current = values[time] + (time < 120 ? progress * (values[time + 1] - values[time]) : 0);
        const points = values.map((a, t) => [t, a]);
        path(c, points, 'init-future');
        path(c, points.slice(0, time + 1).concat([[position, current]]), i ? 'init-path' : 'init-alt');
        if (i) cursor(c, position, current);
      });
      const a = series[time];
      stat(root, 'time', String(time)); stat(root, 'mode', number(a)); stat(root, 'weight', number(Math.pow(a0, 1 / depth)));
      stat(root, 'rate', number(depth * Math.pow(a, 2 - 2 / depth) * (1 - a), 6));
    }
    const api = ArticleSteps.mount(root, {duration: 120, steps: Array.from({length: 121}, (_, i) => ({title: `时间 t = ${i}`, caption: i === 0 ? '各层初始权重都取 a₀^(1/L)，因此两种深度从同一个输出强度开始。' : '输出增大的速度取决于当前乘积 a、深度 L 与剩余误差 1 − a。拖回起点，可以比较最初的学习平台期。'})), render(step, index, progress) { position = index + progress; render(); }});
    [product, depthControl].forEach(control => control.addEventListener('input', () => { api.pause(); render(); }));
    followResize(root, render);
  }
  if (!window.ArticleSteps) return;
  mountScale(); mountCorrelation(); mountSpectrum(); mountDynamics();
})();
