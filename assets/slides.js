// Minimal slide engine: keyboard / click / swipe navigation, step-by-step
// fragments, URL hash per slide, and a `slidechange` event on document so a
// deck can run its own animations:
//
//   document.addEventListener('slidechange', (e) => {
//     if (e.detail.slide.id === 'loop') startLoop();
//   });
(() => {
  const W = 1280;
  const H = 720;
  const stage = document.querySelector('.deck .stage');
  const slides = Array.from(stage.querySelectorAll(':scope > .slide'));
  if (!slides.length) return;

  // Bottom bar: progress, home link, counter.
  const ui = document.createElement('div');
  ui.className = 'deck-ui';
  ui.innerHTML =
    '<div class="deck-progress"></div>' +
    '<div class="deck-bar"><a href="/blog/">← 返回博客</a>' +
    '<span><span class="deck-hint">← → 翻页 · F 全屏　</span><span class="deck-count"></span></span></div>';
  document.body.appendChild(ui);
  const progress = ui.querySelector('.deck-progress');
  const count = ui.querySelector('.deck-count');

  let index = 0;
  let step = 0;

  const fragments = (i) => Array.from(slides[i].querySelectorAll('.fragment'));

  function fit() {
    stage.style.setProperty('--scale', Math.min(innerWidth / W, innerHeight / H));
  }

  function render(prevIndex) {
    slides.forEach((el, i) => {
      el.classList.toggle('is-active', i === index);
      el.classList.toggle('is-past', i < index);
      el.setAttribute('aria-hidden', String(i !== index));
    });
    fragments(index).forEach((f, k) => {
      f.classList.toggle('is-visible', k < step);
      f.classList.toggle('is-current', k === step - 1);
    });
    progress.style.width = ((index + 1) / slides.length) * 100 + '%';
    count.textContent = index + 1 + ' / ' + slides.length;
    history.replaceState(null, '', '#' + (index + 1));
    const detail = { index, step, slide: slides[index] };
    if (prevIndex !== index) {
      document.dispatchEvent(new CustomEvent('slidechange', { detail }));
    } else {
      document.dispatchEvent(new CustomEvent('fragmentchange', { detail }));
    }
  }

  function go(i, atEnd) {
    const prevIndex = index;
    index = Math.max(0, Math.min(slides.length - 1, i));
    step = atEnd ? fragments(index).length : 0;
    render(prevIndex);
  }

  function next() {
    if (step < fragments(index).length) { step++; render(index); }
    else if (index < slides.length - 1) go(index + 1);
  }

  function prev() {
    if (step > 0) { step--; render(index); }
    else if (index > 0) go(index - 1, true);
  }

  document.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target.closest('input, textarea, select, [contenteditable]')) return;
    // Space / Enter on a focused button inside a widget presses the button.
    if ((e.key === ' ' || e.key === 'Enter') && e.target.closest('button, a')) return;
    switch (e.key) {
      case 'ArrowRight': case 'ArrowDown': case 'PageDown': case ' ': case 'Enter':
        e.preventDefault(); next(); break;
      case 'ArrowLeft': case 'ArrowUp': case 'PageUp':
        e.preventDefault(); prev(); break;
      case 'Home': go(0); break;
      case 'End': go(slides.length - 1, true); break;
      case 'f': case 'F':
        if (document.fullscreenElement) document.exitFullscreen();
        else document.documentElement.requestFullscreen?.();
        break;
    }
  });

  // Click the left third to go back, anywhere else to advance. Clicks on
  // links and controls are left alone.
  document.addEventListener('click', (e) => {
    if (e.target.closest('a, button, input, select, textarea, label, [data-interactive]')) return;
    if (Date.now() - lastSwipe < 500) return; // the click a swipe can leave behind
    if (e.clientX < innerWidth / 3) prev(); else next();
  });

  let lastSwipe = 0;
  let touchX = null;
  let touchY = null;
  document.addEventListener('touchstart', (e) => {
    touchX = e.touches[0].clientX;
    touchY = e.touches[0].clientY;
  }, { passive: true });
  document.addEventListener('touchend', (e) => {
    if (touchX === null) return;
    const dx = e.changedTouches[0].clientX - touchX;
    const dy = e.changedTouches[0].clientY - touchY;
    touchX = null;
    if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy)) {
      lastSwipe = Date.now();
      if (dx < 0) next(); else prev();
    }
  });

  addEventListener('resize', fit);
  fit();
  const fromHash = parseInt(location.hash.slice(1), 10);
  index = Number.isFinite(fromHash) ? Math.max(0, Math.min(slides.length - 1, fromHash - 1)) : 0;
  // Wait for the deck's own scripts (which come after this one) to register
  // their slidechange listeners before announcing the first slide.
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => render(-1));
  } else {
    render(-1);
  }
})();
