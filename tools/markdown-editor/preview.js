// markdown-it escapes source HTML. Custom rules also escape all user-provided markup.
(() => {
  'use strict';
  const md = window.markdownit({ html: false, linkify: true, typographer: false, maxNesting: 32 });
  const escape = md.utils.escapeHtml;

  function warn(env, message) {
    if (env.warnings.size < 20) env.warnings.add(message);
  }

  // Register math in the parser so code spans, fenced code, and escaped dollars
  // retain Markdown's own delimiter rules. Whitespace/digit boundaries protect prices.
  function inlineMath(state, silent) {
    const start = state.pos;
    const src = state.src;
    if (src[start] !== '$') return false;
    const delimiter = src[start + 1] === '$' ? '$$' : '$';
    const contentStart = start + delimiter.length;
    if (contentStart >= state.posMax || /\s/.test(src[contentStart])) return false;
    for (let i = contentStart; i < state.posMax; i++) {
      if (src[i] === '\\') { i++; continue; }
      if (src[i] === '\n') return false;
      if (!src.startsWith(delimiter, i)) continue;
      if (i === contentStart || /\s/.test(src[i - 1]) || /\d/.test(src[i + delimiter.length] || '')) return false;
      if (!silent) {
        const token = state.push(delimiter === '$$' ? 'math_display_inline' : 'math_inline', '', 0);
        token.content = src.slice(contentStart, i);
      }
      state.pos = i + delimiter.length;
      return true;
    }
    return false;
  }

  function blockMath(state, start, end, silent) {
    if (state.sCount[start] - state.blkIndent >= 4) return false;
    const line = n => state.src.slice(state.bMarks[n] + state.tShift[n], state.eMarks[n]).trimEnd();
    const first = line(start);
    if (!first.startsWith('$$')) return false;
    const text = first.slice(2);
    let last = start;
    let content;
    if (text.endsWith('$$') && !text.endsWith('\\$$')) {
      content = text.slice(0, -2);
    } else {
      if (text.includes('$$')) return false;
      for (last = start + 1; last < end; last++) {
        const candidate = line(last);
        if (state.sCount[last] < state.blkIndent && candidate) return false;
        if (candidate.endsWith('$$') && !candidate.endsWith('\\$$')) break;
        if (candidate.startsWith('$$') || /^(?:`{3,}|~{3,})/.test(candidate)) return false;
      }
      if (last === end) return false;
      content = text + '\n' + state.getLines(start + 1, last, state.blkIndent, true) + line(last).slice(0, -2);
    }
    if (silent) return true;
    const token = state.push('math_block', '', 0);
    token.block = true;
    token.map = [start, last + 1];
    token.content = content.trim();
    state.line = last + 1;
    return true;
  }

  md.inline.ruler.before('escape', 'math_inline', inlineMath);
  md.block.ruler.before('fence', 'math_block', blockMath, { alt: ['paragraph','reference','blockquote','list'] });

  function math(tokens, i, options, env) {
    const tex = tokens[i].content;
    const display = tokens[i].type !== 'math_inline';
    const tag = tokens[i].type === 'math_block' ? 'div' : 'span';
    if (++env.mathCount > 500 || tex.length > 16000 || !window.katex) {
      warn(env, '部分公式以源码显示');
      return '<code class="md-math-error">' + escape(tex) + '</code>';
    }
    try {
      return '<' + tag + (display ? ' class="md-math-display">' : '>') +
        window.katex.renderToString(tex, { displayMode: display, trust: false, throwOnError: true,
          strict: 'ignore', maxSize: 15, maxExpand: 400 }) + '</' + tag + '>';
    } catch {
      warn(env, '请检查公式 ' + tex.trim().slice(0, 55));
      return '<code class="md-math-error">' + escape(tex) + '</code>';
    }
  }
  md.renderer.rules.math_block = md.renderer.rules.math_inline = md.renderer.rules.math_display_inline = math;

  md.core.ruler.after('inline', 'task_lists', state => {
    const {tokens} = state;
    for (let i = 2; i < tokens.length; i++) {
      if (tokens[i - 2].type !== 'list_item_open' || tokens[i - 1].type !== 'paragraph_open' ||
          tokens[i].type !== 'inline') continue;
      const children = tokens[i].children;
      if (!children.length || children[0].type !== 'text') continue;
      const marker = /^\[([ xX])\]\s+/.exec(children[0].content);
      if (!marker) continue;
      children[0].content = children[0].content.slice(marker[0].length);
      const checkbox = new state.Token('task_box', '', 0);
      checkbox.content = marker[1].toLowerCase() === 'x' ? 'checked' : '';
      children.unshift(checkbox);
      tokens[i - 2].attrJoin('class', 'md-task');
    }
  });
  md.renderer.rules.task_box = (tokens, i) =>
    '<input type="checkbox" disabled aria-label="' + (tokens[i].content ? '已完成' : '待完成') +
    '"' + (tokens[i].content ? ' checked' : '') + '> ';

  md.renderer.rules.table_open = () => '<div class="md-table-scroll"><table>\n';
  md.renderer.rules.table_close = () => '</table></div>\n';

  function safeWebLink(href) {
    if (!/^https?:\/\//i.test(href)) return false;
    try { return Boolean(new URL(href).host); } catch { return false; }
  }
  md.core.ruler.after('inline', 'preview_headings', state => {
    const {tokens, env} = state;
    const suffixes = new Map();
    const plainText = token => token.children ? token.children.map(plainText).join('') :
      (token.type === 'softbreak' || token.type === 'hardbreak' ? ' ' : token.content);
    for (let i = 0; i + 1 < tokens.length; i++) {
      if (tokens[i].type !== 'heading_open') continue;
      const name = plainText(tokens[i + 1]).toLowerCase().trim()
        .replace(/[^\p{L}\p{M}\p{N}_\s-]/gu, '').replace(/\s/g, '-') || 'heading';
      let slug = name;
      let suffix = suffixes.get(name) || 1;
      while (env.headings.has(slug)) slug = name + '-' + suffix++;
      suffixes.set(name, suffix);
      const id = env.prefix + slug;
      env.headings.set(slug, id);
      tokens[i].attrSet('id', id);
    }
  });

  md.renderer.rules.link_open = (tokens, i, opts, env, renderer) => {
    const token = tokens[i];
    const href = token.attrGet('href') || '';
    if (safeWebLink(href)) {
      token.attrSet('rel', 'noopener noreferrer');
      token.attrSet('target', '_blank');
    } else {
      let id;
      if (href.startsWith('#')) {
        try { id = env.headings.get(decodeURIComponent(href.slice(1))); } catch { /* malformed fragment */ }
      }
      if (id) {
        token.attrSet('href', '#' + id);
        token.attrSet('data-md-heading', id);
      } else {
        token.attrs = token.attrs.filter(([name]) => name !== 'href');
      }
    }
    return renderer.renderToken(tokens, i, opts);
  };

  function localImage(src, images) {
    if (!src || /^(?:[a-z][a-z\d+.-]*:|[\\/])/i.test(src)) return '';
    let path = src;
    try { path = decodeURIComponent(src); } catch { return ''; }
    path = path.replaceAll('\\', '/').replace(/^(?:\.\/)+/, '');
    const url = images.get(path) || images.get(path.split('/').pop());
    if (typeof url !== 'string' || !url.startsWith('blob:' + location.origin + '/')) return '';
    return url;
  }
  md.renderer.rules.image = (tokens, i, options, env, renderer) => {
    const token = tokens[i];
    const src = token.attrGet('src') || '';
    const alt = renderer.renderInlineAsText(token.children || [], options, env);
    const url = localImage(src, env.images);
    if (url) return '<img src="' + escape(url) + '" alt="' + escape(alt) + '">';
    if (safeWebLink(src) && /^https:\/\//i.test(src)) {
      if (env.allowRemoteImages) return '<img loading="lazy" referrerpolicy="no-referrer" src="' +
        escape(src) + '" alt="' + escape(alt) + '">';
      warn(env, '可在「文档配图」中开启 HTTPS 网络图片');
    } else if (safeWebLink(src) || src.startsWith('//')) {
      warn(env, '请为网络图片使用完整的 HTTPS 地址');
    } else warn(env, '请选择文档引用的本地配图 ' + src.slice(0, 100));
    return '<span class="md-image-placeholder" title="' + escape(src) + '">配图 · ' +
      escape(alt || src || '未命名') + '</span>';
  };

  const targets = new WeakMap();
  let nextPrefix = 0;
  function prepareTarget(target) {
    if (targets.has(target)) return targets.get(target);
    const prefix = 'md-preview-' + (++nextPrefix) + '-';
    targets.set(target, prefix);
    target.addEventListener('click', event => {
      const link = event.target.closest('a[data-md-heading]');
      if (!link || !target.contains(link)) return;
      event.preventDefault();
      const heading = document.getElementById(link.dataset.mdHeading);
      if (!heading || !target.contains(heading)) return;
      for (let box = target; box && box !== document.body; box = box.parentElement) {
        if (/^(auto|scroll)$/.test(getComputedStyle(box).overflowY)) {
          box.scrollTop += heading.getBoundingClientRect().top - box.getBoundingClientRect().top - 12;
          break;
        }
      }
    });
    return prefix;
  }
  function render(source, target, images = new Map(), {allowRemoteImages = false} = {}) {
    const env = {images, allowRemoteImages, prefix: prepareTarget(target), headings: new Map(),
      warnings: new Set(), mathCount: 0};
    target.innerHTML = md.render(String(source), env);
    return {warnings: Array.from(env.warnings)};
  }
  window.MarkdownPreview = { render };
})();
