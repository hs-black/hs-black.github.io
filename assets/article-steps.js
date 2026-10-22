/* Manual, rewindable figures for articles. A scene is a function of position.
 * ArticleSteps.mount('#figure', {steps: [{title, caption, ...data}, ...],
 *   render(step, index, progress) {}, duration: 1000});
 * progress is the fraction from index to index + 1. Figures start paused.
 */
(() => {
  'use strict';
  const mounted = new WeakMap();
  function mount(target, options) {
    const root = typeof target === 'string' ? document.querySelector(target) : target;
    if (!root) throw new Error('ArticleSteps: figure not found');
    if (mounted.has(root)) return mounted.get(root);
    const steps = options.steps;
    if (!Array.isArray(steps) || !steps.length) throw new Error('ArticleSteps: steps are required');
    const last = steps.length - 1;
    const duration = Math.max(100, Number(options.duration) || 1000);
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    let position = 0, playing = false, frame = 0, previousTime = 0, captionIndex = -1;
    const caption = document.createElement('p');
    caption.className = 'article-step-caption';
    caption.setAttribute('aria-live', 'polite');
    const controls = document.createElement('div');
    controls.className = 'article-step-controls';
    function button(action, text) {
      const el = document.createElement('button');
      el.type = 'button'; el.dataset.action = action; el.textContent = text;
      controls.append(el); return el;
    }
    const prev = button('prev', '上一步');
    const play = button('play', '播放');
    const next = button('next', '下一步');
    const reset = button('reset', '重置');
    const range = document.createElement('input');
    range.type = 'range'; range.min = '0'; range.max = String(last); range.step = '0.01'; range.value = '0';
    range.setAttribute('aria-label', '图解进度');
    const counter = document.createElement('output');
    controls.append(range, counter);
    const figureCaption = root.querySelector(':scope > figcaption');
    root.insertBefore(caption, figureCaption); root.insertBefore(controls, figureCaption);
    function draw(value) {
      position = Math.max(0, Math.min(last, Number.isFinite(Number(value)) ? Number(value) : 0));
      const index = Math.min(last, Math.floor(position + 1e-8));
      const progress = Math.max(0, position - index);
      options.render?.(steps[index], index, progress);
      if (captionIndex !== index) {
        captionIndex = index;
        caption.replaceChildren();
        if (steps[index].title) {
          const title = document.createElement('strong');
          title.textContent = steps[index].title + '。'; caption.append(title, ' ');
        }
        caption.append(String(steps[index].caption || ''));
      }
      range.value = String(position);
      range.setAttribute('aria-valuetext', (index + 1) + ' / ' + steps.length + '：' + (steps[index].title || ''));
      counter.textContent = (index + 1) + ' / ' + steps.length;
      prev.disabled = position <= 0; next.disabled = position >= last;
      play.disabled = last === 0;
      play.textContent = playing ? '暂停' : (position >= last ? '重播' : '播放');
      play.setAttribute('aria-pressed', String(playing));
    }
    function pause() {
      playing = false; cancelAnimationFrame(frame); previousTime = 0;
      draw(position);
    }
    function seek(value) { pause(); draw(value); }
    function tick(now) {
      if (!playing) return;
      if (previousTime) draw(position + Math.min(250, now - previousTime) / duration);
      previousTime = now;
      if (position >= last) pause(); else frame = requestAnimationFrame(tick);
    }
    prev.addEventListener('click', () => seek(Math.ceil(position - 1e-8) - 1));
    next.addEventListener('click', () => seek(Math.floor(position + 1e-8) + 1));
    reset.addEventListener('click', () => seek(0));
    range.addEventListener('input', () => seek(range.value));
    play.addEventListener('click', () => {
      if (playing) { pause(); return; }
      if (position >= last) seek(0);
      if (reduced.matches) { seek(Math.min(last, Math.floor(position) + 1)); return; }
      playing = true; previousTime = 0; draw(position); frame = requestAnimationFrame(tick);
    });
    document.addEventListener('visibilitychange', () => { if (document.hidden && playing) pause(); });
    const observer = new IntersectionObserver(entries => {
      if (!entries[0].isIntersecting && playing) pause();
    });
    observer.observe(root);
    const api = { seek, pause, render: () => draw(position), get position() { return position; }, get playing() { return playing; } };
    mounted.set(root, api); draw(0); return api;
  }
  window.ArticleSteps = { mount };
})();
