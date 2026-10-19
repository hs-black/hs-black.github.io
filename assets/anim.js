// Timeline animation for decks and posts, in the spirit of Remotion: the
// whole scene is a pure function of time t, so it can play, rewind, scrub and
// jump to any step and always look the same.
//
//   const tl = Anim.timeline();
//   tl.to('#tok', { x: 200, fill: 'var(--accent)' }, { dur: 0.8 })   // after the previous tween
//     .to('.blk', { opacity: 1, y: 0 }, { at: '<', stagger: 0.05 })    // together with the previous one
//     .label('step')                                                   // a stop for the next key press
//     .to('#arrow', { draw: 1 })                                       // stroke-draw a path or line
//     .to('#n', { text: 4096, fmt: (v) => Math.round(v) + ' tok' })
//     .during((p) => drawTrail(p), { at: 0, dur: 3 });                  // custom per-frame drawing
//   Anim.slide(document.getElementById('my-slide'), tl, { steps: true, controls: true });
//
// Props: x, y, scale, rotate (deg), opacity, fill, stroke, color, background
// (any CSS color incl. var(--token)), draw (0..1 of path length), text (a
// number tweened through fmt), attr: { width: 10, cx: 3, ... } for numeric SVG
// attributes, and style: { '--p': 1 } for numeric CSS properties.
// at: number (seconds) | '+=0.2' after the end of the timeline so far | '<' /
// '<+=0.2' relative to the previous tween's start | a label name. Default: at
// the end of the timeline so far (so tweens play one after another).
(() => {
  const EASE = {
    linear: (t) => t,
    in: (t) => t * t * t,
    out: (t) => 1 - (1 - t) ** 3,
    inOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2),
    back: (t) => { const c = 1.70158; return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2; },
  };
  const COLOR_PROPS = ['fill', 'stroke', 'color', 'background'];
  const XFORM = ['x', 'y', 'scale', 'rotate'];
  const reduce = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Resolve any CSS color (including var(--x)) to [r, g, b, a].
  let probe;
  function rgba(c) {
    if (!probe) { probe = document.createElement('i'); probe.style.display = 'none'; document.body.appendChild(probe); }
    probe.style.color = '';
    probe.style.color = c;
    const c2 = getComputedStyle(probe).color;
    const m = c2.match(/[\d.]+/g) || [0, 0, 0];
    // color-mix() and friends compute to color(srgb r g b / a) with channels in 0..1.
    const k = c2.startsWith('color(') ? 255 : 1;
    return [m[0] * k, m[1] * k, m[2] * k, m[3] === undefined ? 1 : +m[3]];
  }

  function targets(sel, root) {
    if (typeof sel === 'string') return Array.from((root || document).querySelectorAll(sel));
    if (sel instanceof Element) return [sel];
    return Array.from(sel).flatMap((x) => targets(x, root));
  }

  // Per-element animated state, flushed to the DOM after every render.
  const STATE = new WeakMap();
  function state(el) {
    let s = STATE.get(el);
    if (!s) {
      s = { x: 0, y: 0, scale: 1, rotate: 0 };
      STATE.set(el, s);
      if (el instanceof SVGElement) { el.style.transformBox = 'fill-box'; el.style.transformOrigin = 'center'; }
    }
    return s;
  }

  function initial(el, key) {
    if (XFORM.includes(key)) return state(el)[key];
    if (key === 'opacity') { const o = getComputedStyle(el).opacity; return o === '' ? 1 : +o; }
    if (key === 'draw') return el.dataset.draw !== undefined ? +el.dataset.draw : 0;
    if (key === 'text') { if (el.dataset.animText === undefined) el.dataset.animText = el.textContent; return parseFloat(el.textContent) || 0; }
    if (COLOR_PROPS.includes(key)) {
      const cs = getComputedStyle(el);
      return rgba(key === 'background' ? cs.backgroundColor : cs[key]);
    }
    if (key.startsWith('attr:')) return parseFloat(el.getAttribute(key.slice(5))) || 0;
    if (key.startsWith('style:')) return parseFloat(getComputedStyle(el).getPropertyValue(key.slice(6))) || 0;
    return 0;
  }

  function apply(el, key, v, fmt) {
    if (XFORM.includes(key)) { state(el)[key] = v; state(el).dirty = true; return; }
    if (key === 'opacity') { el.style.opacity = v; return; }
    if (key === 'text') { el.textContent = fmt ? fmt(v) : String(Math.round(v)); return; }
    if (key === 'draw') {
      const len = el.getTotalLength ? el.getTotalLength() : 0;
      el.style.strokeDasharray = len + ' ' + len;
      el.style.strokeDashoffset = String(len * (1 - v));
      return;
    }
    if (COLOR_PROPS.includes(key)) {
      const c = 'rgba(' + v.map((x, i) => (i < 3 ? Math.round(x) : x.toFixed(3))).join(',') + ')';
      if (key === 'background') el.style.backgroundColor = c; else el.style[key] = c;
      return;
    }
    if (key.startsWith('attr:')) { el.setAttribute(key.slice(5), v); return; }
    if (key.startsWith('style:')) { el.style.setProperty(key.slice(6), v); }
  }

  const lerp = (a, b, t) => (Array.isArray(a) ? a.map((x, i) => x + (b[i] - x) * t) : a + (b - a) * t);

  function timeline(opts = {}) {
    const root = opts.root || document;
    const tweens = []; // { el, key, to, start, dur, ease, fmt }
    const labels = []; // [{ name, t }]
    const calls = [];  // [{ t, fn }]
    const hooks = [];  // during(): [{ fn, start, dur, ease }]
    let prevStart = 0;
    let end = 0;       // end of the timeline so far
    let built = null;  // per (el,key) sorted tween lists with resolved from-values

    function resolveAt(at) {
      if (at === undefined) return end;
      if (typeof at === 'number') return at;
      if (at.startsWith('+=')) return end + parseFloat(at.slice(2));
      if (at.startsWith('<')) return prevStart + (at.length > 1 ? parseFloat(at.slice(3)) : 0);
      const l = labels.find((x) => x.name === at);
      return l ? l.t : end;
    }

    const api = {
      to(sel, props, o = {}) {
        const els = targets(sel, root);
        const start = resolveAt(o.at);
        const dur = o.dur ?? 0.6;
        const stagger = o.stagger ?? 0;
        const ease = typeof o.ease === 'function' ? o.ease : EASE[o.ease || 'inOut'];
        els.forEach((el, i) => {
          const s = start + i * stagger;
          for (const [k, v] of Object.entries(props)) {
            if (k === 'fmt') continue;
            if (k === 'attr' || k === 'style') {
              for (const [a, av] of Object.entries(v)) tweens.push({ el, key: k + ':' + a, to: av, start: s, dur, ease });
            } else {
              tweens.push({ el, key: k, to: v, start: s, dur, ease, fmt: props.fmt });
            }
          }
        });
        prevStart = start;
        end = Math.max(end, start + dur + Math.max(0, els.length - 1) * stagger);
        built = null;
        return api;
      },
      set(sel, props, o = {}) { return api.to(sel, props, { ...o, dur: 0 }); },
      // Run fn(progress, t) on every render once t reaches the start, with
      // progress eased over dur (clamped to 1 after). For custom drawing such
      // as trails or curves; fn must draw purely from its arguments.
      during(fn, o = {}) {
        const start = resolveAt(o.at);
        const dur = o.dur ?? 0.6;
        hooks.push({ fn, start, dur, ease: typeof o.ease === 'function' ? o.ease : EASE[o.ease || 'linear'] });
        prevStart = start;
        end = Math.max(end, start + dur);
        return api;
      },
      wait(sec) { prevStart = end; end += sec; return api; },
      label(name, at) { const t = resolveAt(at); labels.push({ name, t }); end = Math.max(end, t); return api; },
      call(fn, at) { calls.push({ t: resolveAt(at), fn }); return api; },
      get duration() { return end; },
      get labels() { return labels.slice().sort((a, b) => a.t - b.t); },
      render,
      player: null,
    };

    function build() {
      const groups = new Map();
      for (const w of tweens) {
        let m = groups.get(w.el);
        if (!m) { m = new Map(); groups.set(w.el, m); }
        if (!m.has(w.key)) m.set(w.key, []);
        m.get(w.key).push(w);
      }
      built = [];
      for (const [el, m] of groups) {
        for (const [key, list] of m) {
          list.sort((a, b) => a.start - b.start);
          const init = initial(el, key);
          let from = init;
          const segs = list.map((w) => {
            const to = COLOR_PROPS.includes(key) ? rgba(w.to) : w.to;
            const seg = { ...w, from, to };
            from = to;
            return seg;
          });
          built.push({ el, key, init, segs });
        }
      }
    }

    let lastT = -1;
    function render(t) {
      if (!built) build();
      for (const { el, key, init, segs } of built) {
        // Before its first tween a text element keeps whatever it said originally.
        if (key === 'text' && segs[0] && t < segs[0].start) { el.textContent = el.dataset.animText; continue; }
        let v = init;
        let fmt;
        for (const s of segs) {
          if (t < s.start) break;
          fmt = s.fmt;
          const p = s.dur === 0 ? 1 : Math.min(1, (t - s.start) / s.dur);
          v = lerp(s.from, s.to, s.ease(p));
        }
        if (fmt === undefined && segs[0]) fmt = segs[0].fmt;
        apply(el, key, v, fmt);
      }
      for (const { el } of built) {
        const s = STATE.get(el);
        if (s && s.dirty) {
          el.style.transform = 'translate(' + s.x + 'px,' + s.y + 'px) scale(' + s.scale + ') rotate(' + s.rotate + 'deg)';
          s.dirty = false;
        }
      }
      for (const h of hooks) {
        if (t >= h.start) h.fn(h.ease(h.dur === 0 ? 1 : Math.min(1, (t - h.start) / h.dur)), t);
      }
      // Fire calls crossed while moving forward.
      if (t > lastT) calls.forEach((c) => { if (c.t > lastT && c.t <= t) c.fn(); });
      lastT = t;
    }

    // ---------- player ----------
    let t = 0;
    let target = 0;
    let raf = 0;
    let lastFrame = 0;
    const listeners = [];
    const player = {
      get time() { return t; },
      seek(x) { cancelAnimationFrame(raf); raf = 0; t = target = Math.max(0, Math.min(api.duration, x)); render(t); emit(); },
      playTo(x) {
        target = Math.max(0, Math.min(api.duration, x));
        if (reduce() || target < t) { player.seek(target); return; }
        if (!raf) { lastFrame = performance.now(); raf = requestAnimationFrame(tick); }
      },
      play() { player.playTo(api.duration); },
      pause() { cancelAnimationFrame(raf); raf = 0; target = t; emit(); },
      get playing() { return !!raf; },
      onupdate(fn) { listeners.push(fn); },
      loop: !!opts.loop,
    };
    function emit() { listeners.forEach((f) => f(t)); }
    function tick(now) {
      const dt = Math.min(0.1, (now - lastFrame) / 1000);
      lastFrame = now;
      t = Math.min(target, t + dt * (opts.speed || 1));
      render(t);
      if (t < target) raf = requestAnimationFrame(tick);
      else if (player.loop && target >= api.duration) { t = 0; lastT = -1; raf = requestAnimationFrame(tick); }
      else raf = 0;
      emit();
    }
    api.player = player;
    // Colors are resolved from CSS tokens when the timeline is built; rebuild
    // when the color scheme flips so var(--accent) etc. follow light/dark.
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
      if (!built) return;
      built.forEach((b) => { if (b.key === 'text' && b.el.dataset.animText !== undefined) b.el.textContent = b.el.dataset.animText; });
      built = null; lastT = t; render(t);
    });
    return api;
  }

  // A small play / scrub bar.
  function controls(tl, container) {
    const bar = document.createElement('div');
    bar.className = 'anim-bar';
    bar.setAttribute('data-interactive', '');
    bar.innerHTML = '<button type="button" class="anim-play" aria-label="播放">▶</button>' +
      '<input type="range" min="0" max="1000" value="0" aria-label="时间轴">' +
      '<span class="anim-time"></span>';
    container.appendChild(bar);
    const btn = bar.querySelector('button');
    const range = bar.querySelector('input');
    const time = bar.querySelector('.anim-time');
    const p = tl.player;
    btn.addEventListener('click', () => {
      if (p.playing) p.pause();
      else { if (p.time >= tl.duration - 1e-3) p.seek(0); p.play(); }
    });
    range.addEventListener('input', () => p.seek((range.value / 1000) * tl.duration));
    p.onupdate((x) => {
      range.value = String(Math.round((x / (tl.duration || 1)) * 1000));
      time.textContent = x.toFixed(1) + ' / ' + tl.duration.toFixed(1) + 's';
      btn.textContent = p.playing ? '❚❚' : '▶';
      btn.setAttribute('aria-label', p.playing ? '暂停' : (x >= tl.duration ? '重新播放' : '播放'));
    });
    return bar;
  }

  // Bind a timeline to a deck slide. steps: each label is a stop reached by
  // one key press (hidden .fragment elements are added for slides.js).
  // Otherwise the whole timeline plays when the slide is entered.
  function slide(slideEl, tl, o = {}) {
    const stops = tl.labels.map((l) => l.t);
    if (!stops.length || stops[stops.length - 1] < tl.duration) stops.push(tl.duration);
    let startTimer = 0;
    if (o.steps) {
      // The first stop plays on entry; only later stops need a key press.
      for (let i = 1; i < stops.length; i++) {
        const f = document.createElement('span');
        f.className = 'fragment anim-step';
        f.hidden = true;
        slideEl.appendChild(f);
      }
    }
    if (o.controls) controls(tl, o.controlsIn || slideEl);
    tl.player.loop = !!o.loop;
    tl.player.seek(0);
    if (o.steps) tl.player.onupdate((time) => {
      if (!slideEl.classList.contains('is-active') || tl.player.playing) return;
      clearTimeout(startTimer);
      const step = time >= tl.duration - 1e-3 ? stops.length - 1 :
        Math.max(-1, stops.filter((t) => t <= time + 1e-3).length - 1);
      document.dispatchEvent(new CustomEvent('animseek', { detail: { slide: slideEl, step } }));
    });
    const stopFor = (step) => (o.steps ? (step < stops.length ? stops[step] : tl.duration) : tl.duration);
    document.addEventListener('slidechange', (e) => {
      clearTimeout(startTimer);
      if (e.detail.slide !== slideEl) { tl.player.pause(); return; }
      const end = stopFor(e.detail.step);
      if (e.detail.step > 0 && o.steps) tl.player.seek(stops[e.detail.step - 1]); else tl.player.seek(0);
      startTimer = setTimeout(() => {
        if (slideEl.classList.contains('is-active')) tl.player.playTo(end);
      }, o.delay ?? 300);
    });
    document.addEventListener('fragmentchange', (e) => {
      if (e.detail.slide !== slideEl || !o.steps) return;
      const end = stopFor(e.detail.step);
      if (end < tl.player.time) tl.player.seek(end); else tl.player.playTo(end);
    });
    return tl;
  }

  // For articles: play when scrolled into view (once), with controls.
  function inView(el, tl, o = {}) {
    if (o.controls !== false) controls(tl, o.controlsIn || el);
    tl.player.loop = !!o.loop;
    tl.player.seek(0);
    const io = new IntersectionObserver((es) => {
      es.forEach((x) => { if (x.isIntersecting) { tl.player.play(); if (!o.loop) io.disconnect(); } else if (o.loop) tl.player.pause(); });
    }, { threshold: 0.4 });
    io.observe(el);
    return tl;
  }

  window.Anim = { timeline, slide, inView, controls, ease: EASE };
})();
