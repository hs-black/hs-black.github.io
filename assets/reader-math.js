/* Scoped symbol explanations for mathematical articles.
 *
 * Load KaTeX, its auto-render helper, and reader-math.css before this file.
 * Formula: \htmlData{symbol=batch}{B}
 * Definitions:
 * <script type="application/json" id="math-definitions">
 * {"global":{"batch":{"label":"B · batch 大小","description":"本轮参与计算的请求数。","unit":"条请求"}},
 *  "scopes":{"prefill":{"batch":{"label":"B · token 数","description":"本次矩阵乘处理的 token 数。"}}}}
 * </script>
 * Set data-math-scope="prefill" on the enclosing section when a symbol changes
 * meaning. Keys are explicit: no matching or guessing based on rendered letters.
 * Keep the essential definitions in the prose beside the equation as well.
 *
 * Dynamic HTML labels: <span data-tex="\\frac{a}{b}"></span>;
 * call ReaderMath.render(container) after changing their data-tex value.
 * SVG <text> is deliberately excluded. For a typeset SVG label, author a sized
 * <foreignObject> containing an XHTML element with data-tex, or put the formula
 * next to the diagram in ordinary HTML.
 */
(() => {
  'use strict';
  const HTML_NS = 'http://www.w3.org/1999/xhtml';
  const KEY = /^[a-zA-Z][a-zA-Z0-9_-]{0,79}$/;
  const TIP_ID = 'reader-symbol-tip';
  let definitions = { global: {}, scopes: {} };
  let tip, tipTitle, tipBody, tipUnit;
  let active = null;
  let pinned = false;
  let hideTimer;
  const known = new WeakMap();

  const all = (root, selector) => [
    ...(root.nodeType === 1 && root.matches(selector) ? [root] : []),
    ...root.querySelectorAll(selector),
  ];

  function readDefinitions() {
    const source = document.getElementById('math-definitions');
    if (!source) return;
    try {
      const data = JSON.parse(source.textContent);
      if (data && typeof data === 'object' && !Array.isArray(data)) {
        definitions = { global: data.global || {}, scopes: data.scopes || {} };
      }
    } catch (_) {
      // A malformed author manifest must not erase the article or its equations.
      source.dataset.mathDefinitionsError = 'invalid-json';
    }
  }

  function definitionFor(el) {
    const key = el.dataset.symbol;
    if (!KEY.test(key || '')) return null;
    // Walk outwards: a nested scope may override one symbol and inherit others.
    let scope = el.closest('[data-math-scope]');
    let value;
    while (scope) {
      const entries = definitions.scopes[scope.dataset.mathScope];
      if (entries && Object.prototype.hasOwnProperty.call(entries, key)) {
        value = entries[key];
        break;
      }
      scope = scope.parentElement?.closest('[data-math-scope]');
    }
    if (value === undefined && Object.prototype.hasOwnProperty.call(definitions.global, key)) {
      value = definitions.global[key];
    }
    if (typeof value === 'string') return { label: el.textContent.trim(), description: value, unit: '' };
    if (!value || typeof value !== 'object' || !value.description) return null;
    return {
      label: String(value.label || el.textContent.trim()),
      description: String(value.description),
      unit: value.unit ? String(value.unit) : '',
    };
  }

  function makeTip() {
    if (tip) return;
    tip = document.createElement('div');
    tip.id = TIP_ID;
    tip.className = 'reader-symbol-tip';
    tip.setAttribute('role', 'tooltip');
    tip.hidden = true;
    tipTitle = document.createElement('strong');
    tipBody = document.createElement('p');
    tipUnit = document.createElement('small');
    tip.append(tipTitle, tipBody, tipUnit);
    document.body.append(tip);
    tip.addEventListener('pointerenter', () => clearTimeout(hideTimer));
    tip.addEventListener('pointerleave', () => scheduleHide());
  }

  function describedBy(el, enabled) {
    const ids = new Set((el.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean));
    enabled ? ids.add(TIP_ID) : ids.delete(TIP_ID);
    ids.size ? el.setAttribute('aria-describedby', [...ids].join(' ')) : el.removeAttribute('aria-describedby');
  }

  function positionTip() {
    if (!active || tip.hidden) return;
    if (!active.isConnected) { hide(); return; }
    const box = active.getBoundingClientRect();
    const viewport = window.visualViewport;
    const left = viewport?.offsetLeft || 0;
    const top = viewport?.offsetTop || 0;
    const width = viewport?.width || innerWidth;
    const height = viewport?.height || innerHeight;
    const margin = 10;
    if (box.bottom < top || box.top > top + height || box.right < left || box.left > left + width) { hide(); return; }
    tip.style.maxHeight = Math.max(1, height - 2 * margin) + 'px';
    tip.style.maxWidth = Math.max(1, Math.min(360, width - 2 * margin)) + 'px';
    const rect = tip.getBoundingClientRect();
    const x = Math.max(left + margin, Math.min(left + width - rect.width - margin,
      box.left + box.width / 2 - rect.width / 2));
    const above = box.top - rect.height - 9;
    const below = box.bottom + 9;
    const y = above >= top + margin ? above
      : Math.max(top + margin, Math.min(below, top + height - rect.height - margin));
    tip.style.left = Math.round(x) + 'px';
    tip.style.top = Math.round(y) + 'px';
  }

  function hide() {
    clearTimeout(hideTimer);
    if (active) {
      active.classList.remove('reader-symbol-active');
      active.setAttribute('aria-expanded', 'false');
      describedBy(active, false);
    }
    active = null;
    pinned = false;
    if (tip) tip.hidden = true;
  }

  function show(el, pin = false) {
    const value = definitionFor(el);
    if (!value) return;
    clearTimeout(hideTimer);
    if (active && active !== el) hide();
    makeTip();
    active = el;
    pinned = pin;
    tipTitle.textContent = value.label;
    tipBody.textContent = value.description;
    tipUnit.textContent = value.unit ? '单位：' + value.unit : '';
    tipUnit.hidden = !value.unit;
    tip.hidden = false;
    el.classList.add('reader-symbol-active');
    el.setAttribute('aria-expanded', 'true');
    describedBy(el, true);
    positionTip();
  }

  function scheduleHide() {
    clearTimeout(hideTimer);
    if (pinned || active === document.activeElement) return;
    hideTimer = setTimeout(hide, 160);
  }

  function toggle(el) {
    if (active === el && pinned) hide();
    else show(el, true);
  }

  function accessibleMath(html) {
    // KaTeX normally hides its visual HTML from assistive technology because the
    // sibling MathML already represents the equation. Keep that MathML intact,
    // and expose only the explicitly annotated symbol controls in the HTML tree.
    // A focusable element must never remain inside an aria-hidden ancestor.
    const visit = el => {
      if (el.classList.contains('reader-symbol')) {
        el.removeAttribute('aria-hidden');
        [...el.children].forEach(child => child.setAttribute('aria-hidden', 'true'));
      } else if (el.querySelector('.reader-symbol')) {
        el.removeAttribute('aria-hidden');
        [...el.children].forEach(visit);
      } else {
        el.setAttribute('aria-hidden', 'true');
      }
    };
    visit(html);
  }

  function enhance(root) {
    for (const el of all(root, '[data-symbol]')) {
      if (el.namespaceURI !== HTML_NS || el.closest('.katex-mathml')) continue;
      const value = definitionFor(el);
      if (!value) continue;
      el.classList.add('reader-symbol');
      el.setAttribute('role', 'button');
      el.setAttribute('tabindex', '0');
      el.setAttribute('aria-label', value.label + '：' + value.description +
        (value.unit ? '；单位：' + value.unit : ''));
      el.setAttribute('aria-expanded', active === el ? 'true' : 'false');
      if (known.has(el)) continue;
      known.set(el, true);
      el.addEventListener('pointerenter', e => {
        if (e.pointerType === 'touch') return;
        if (active === el && pinned) return;
        show(el);
      });
      el.addEventListener('pointerleave', scheduleHide);
      el.addEventListener('focus', () => show(el, active === el && pinned));
      el.addEventListener('blur', scheduleHide);
      el.addEventListener('click', e => { e.preventDefault(); e.stopPropagation(); toggle(el); });
      el.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault(); e.stopPropagation(); toggle(el);
        }
      });
    }
    for (const html of all(root, '.katex-html')) {
      if (html.querySelector('.reader-symbol')) accessibleMath(html);
    }
  }

  const options = {
    throwOnError: false,
    strict: errorCode => errorCode === 'htmlExtension' ? 'ignore' : 'warn',
    trust: context => context.command === '\\htmlData' &&
      Object.keys(context.attributes || {}).length === 1 &&
      KEY.test(context.attributes['data-symbol'] || ''),
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '\\[', right: '\\]', display: true },
      { left: '\\(', right: '\\)', display: false },
    ],
    ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option', 'svg', 'math'],
    ignoredClasses: ['katex', 'no-math', 'reader-symbol-tip', 'math-source'],
  };

  function render(root = document.body) {
    if (root?.nodeType === 9) root = root.body;
    if (!root || !root.querySelectorAll) return { available: false, errors: 0 };
    if (active && !active.isConnected) hide();
    readDefinitions();
    const available = typeof window.katex?.render === 'function';
    if (available) {
      for (const el of all(root, '[data-tex]')) {
        if (el.namespaceURI !== HTML_NS || el.closest('script, style, pre, code')) continue;
        const tex = el.dataset.tex;
        const displayMode = el.dataset.display === 'true';
        const stamp = (displayMode ? 'block:' : 'inline:') + tex;
        if (el.dataset.mathRendered === stamp && el.querySelector('.katex')) continue;
        window.katex.render(tex, el, { ...options, displayMode });
        el.dataset.mathRendered = stamp;
      }
      if (root.namespaceURI === HTML_NS && typeof window.renderMathInElement === 'function') {
        window.renderMathInElement(root, options);
      }
    } else {
      // Preserve meaningful text if a script cannot load, including empty
      // explicit HTML labels. A later render() call can safely retry.
      for (const el of all(root, '[data-tex]')) {
        if (el.namespaceURI === HTML_NS && !el.textContent.trim()) el.textContent = el.dataset.tex;
      }
    }
    enhance(root);
    const result = { available, errors: root.querySelectorAll('.katex-error').length };
    document.dispatchEvent(new CustomEvent('readermathrender', { detail: result }));
    return result;
  }

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && active) { e.preventDefault(); hide(); }
  });
  document.addEventListener('pointerdown', e => {
    if (active && !active.contains(e.target) && !tip.contains(e.target)) hide();
  });
  document.addEventListener('scroll', positionTip, { capture: true, passive: true });
  addEventListener('resize', positionTip);
  window.visualViewport?.addEventListener('resize', positionTip);
  window.visualViewport?.addEventListener('scroll', positionTip);
  window.ReaderMath = { render, hide };
  const start = () => render(document.body);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
