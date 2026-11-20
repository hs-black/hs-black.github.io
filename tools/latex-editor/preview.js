// Lightweight, local reading preview. TeX input is never evaluated as JavaScript or HTML.
(() => {
  'use strict';

  const make = (tag, className) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    return element;
  };

  // Preserve literal percent signs in control symbols, inline code and verbatim blocks.
  function stripComments(source) {
    return source.replace(/\\begin\{(verbatim\*?|lstlisting|minted)\}[\s\S]*?\\end\{\1\}|\\verb\*?([^\w\s])[^\n]*?\2|\\(?:[a-zA-Z@]+|[\s\S])|%[^\n]*(?:\n|$)/g,
      part => part[0] === '%' ? '' : part);
  }

  class Cursor {
    constructor(source, warn) {
      this.source = source;
      this.i = 0;
      this.warn = warn;
    }
    take(pattern) {
      pattern.lastIndex = this.i;
      const match = pattern.exec(this.source);
      if (!match) return '';
      this.i = pattern.lastIndex;
      return match[0];
    }
    group(optional = false) {
      const open = optional ? '[' : '{';
      const close = optional ? ']' : '}';
      const old = this.i;
      this.take(/\s*/y);
      if (this.source[this.i] !== open) {
        this.i = old;
        return null;
      }
      const start = ++this.i;
      let depth = 1;
      let braces = 0;
      while (this.i < this.source.length) {
        const char = this.source[this.i++];
        if (char === '\\') { this.i = Math.min(this.source.length, this.i + 1); continue; }
        if (optional && char === '{') braces++;
        if (optional && char === '}') braces--;
        if (braces) continue;
        if (char === open) depth++;
        if (char === close && --depth === 0) return this.source.slice(start, this.i - 1);
      }
      this.warn('有一组 ' + open + ' 缺少配对的 ' + close);
      return this.source.slice(start);
    }
    math(end) {
      const start = this.i;
      while (this.i < this.source.length) {
        if (this.source.startsWith(end, this.i)) {
          const body = this.source.slice(start, this.i);
          this.i += end.length;
          return body;
        }
        this.i += this.source[this.i] === '\\' ? 2 : 1;
      }
      this.warn('公式缺少结束符 ' + end);
      return this.source.slice(start);
    }
    environment(name) {
      const start = this.i;
      if (/^(verbatim\*?|lstlisting|minted)$/.test(name)) {
        const close = '\\end{' + name + '}';
        const pos = this.source.indexOf(close, start);
        if (pos >= 0) { this.i = pos + close.length; return this.source.slice(start, pos); }
      }
      const commands = /\\(?:\\|(?:(begin|end)\s*\{([^{}]*)\}))/g;
      commands.lastIndex = start;
      let depth = 1;
      for (let match; (match = commands.exec(this.source));) {
        if (match[2] !== name) continue;
        depth += match[1] === 'begin' ? 1 : -1;
        if (!depth) { this.i = commands.lastIndex; return this.source.slice(start, match.index); }
      }
      this.warn('环境 ' + name + ' 缺少对应的结束标记');
      this.i = this.source.length;
      return this.source.slice(start);
    }
  }

  function render(source, target) {
    const warnings = new Set();
    const warn = message => { if (warnings.size < 20) warnings.add(message); };
    const fragment = document.createDocumentFragment();
    const meta = new Map();
    const macros = {}; // KaTeX expects a plain macro dictionary. Names are validated control sequences.
    const textMacros = new Map();
    let mathCount = 0;
    let steps = 0;
    let truncated = false;
    let expandedTextLength = 0;
    const scopes = [0, 0, 0];
    const symbols = { LaTeX: 'LaTeX', TeX: 'TeX', and: '　', ldots: '…', dots: '…',
      textbackslash: '∖', textasciitilde: '~', textasciicircum: '^', quad: '　', qquad: '　　' };
    const formatting = { textbf: 'strong', textit: 'em', emph: 'em', texttt: 'code', underline: 'u',
      textrm: 'span', textsf: 'span', textnormal: 'span', mbox: 'span', textsuperscript: 'sup', textsubscript: 'sub' };
    const mathEnvs = /^(equation|align|alignat|flalign|gather|multline|displaymath|math)\*?$/;
    function formula(tex, display) {
      const element = make(display ? 'div' : 'span', display ? 'tex-math-display' : 'tex-math-inline');
      mathCount++;
      const limited = tex.length > 16000 || mathCount > 500;
      if (!window.katex || limited) {
        element.textContent = tex;
        warn(limited ? '较长的公式或第 500 条以后的公式以源码显示' : '公式组件尚未加载');
        return element;
      }
      try {
        window.katex.render(tex, element, { displayMode: display, trust: false, throwOnError: true,
          strict: 'ignore', maxExpand: 400, maxSize: 15, macros, output: 'htmlAndMathml' });
      } catch (error) {
        element.classList.add('tex-math-error');
        element.textContent = tex;
        element.title = String(error.message).slice(0, 240);
        warn('公式语法或命令需要检查：' + tex.trim().slice(0, 45));
      }
      return element;
    }

    function flow(source, host, depth = 0, inlineOnly = false, list = false) {
      if (depth > 32) { warn('嵌套超过 32 层，深层内容以源码显示'); host.append(source); return; }
      const c = new Cursor(source, warn);
      let current = host;
      let paragraph = null;
      function inline(node) {
        if (inlineOnly) { current.append(node); return; }
        if (!paragraph) {
          if (typeof node === 'string' && !node.trim()) return;
          paragraph = make('p');
          current.append(paragraph);
        }
        paragraph.append(node);
      }
      function block(element) { paragraph = null; current.append(element); }
      function unsupported(name, label) {
        warn('简版预览保留了 ' + name + ' 的源码');
        const el = make('code', 'tex-unsupported');
        el.textContent = label || name;
        inline(el);
      }
      function defineMacro(cmd) {
        const name = c.group() || c.take(/\\[a-zA-Z@]+/y);
        const arity = c.group(true);
        const defaultArg = c.group(true);
        const body = c.group();
        if (!/^\\[a-zA-Z@]+$/.test(name || '') || body === null) { warn('宏定义需要名称和替换内容'); return; }
        if (defaultArg !== null || body.length > 1000) { warn(name + ' 的宏定义需要完整 LaTeX 编译'); return; }
        if (cmd === 'providecommand' && Object.hasOwn(macros, name)) return;
        if (textMacros.size >= 200) { warn('简版预览最多加载 200 个文档宏'); return; }
        macros[name] = body;
        textMacros.set(name, { arity: Number(arity || 0), body });
      }
      function table(body) {
        const table = make('table');
        const wrap = make('div', 'tex-table-wrap');
        wrap.append(table);
        const rows = new Cursor(body, warn);
        rows.group(true); rows.group(); // Position and column description.
        // Read grouping, math, and escaped separators before splitting rows/cells.
        let row = make('tr'), cell = '';
        function finishCell() {
          steps++;
          const td = make('td'); flow(cell, td, depth + 1, true); row.append(td); cell = '';
        }
        const clean = rows.source.slice(rows.i).replace(/\\(?:toprule|midrule|bottomrule|hline)\b/g, '');
        const tokens = new Cursor(clean, warn);
        while (tokens.i < clean.length) {
          if (steps > 60000) { truncated = true; break; }
          if (clean.startsWith('\\\\', tokens.i)) {
            tokens.i += 2; tokens.group(true); finishCell(); table.append(row); row = make('tr'); continue;
          }
          const ch = clean[tokens.i];
          if (ch === '&') { tokens.i++; finishCell(); continue; }
          if (ch === '{') { cell += '{' + tokens.group() + '}'; continue; }
          if (ch === '$') { tokens.i++; cell += '$' + tokens.math('$') + '$'; continue; }
          if (ch === '\\') { cell += clean.slice(tokens.i, tokens.i + 2); tokens.i += 2; continue; }
          cell += ch; tokens.i++;
        }
        if (cell.trim() || row.childElementCount) { finishCell(); table.append(row); }
        return wrap;
      }
      while (c.i < source.length) {
        if (++steps > 60000) { truncated = true; return; }
        if (source[c.i] === '$' || source.startsWith('\\(', c.i) || source.startsWith('\\[', c.i)) {
          const open = c.take(/\$\$?|\\[([]/y);
          const display = open === '$$' || open === '\\[';
          const end = open === '\\(' ? '\\)' : open === '\\[' ? '\\]' : open;
          const el = formula(c.math(end), display);
          if (display) block(el); else inline(el);
          continue;
        }
        if (source[c.i] === '{') {
          const span = make('span'); inline(span); flow(c.group(), span, depth + 1, true); continue;
        }
        if (source[c.i] !== '\\') {
          const raw = c.take(/[^\\${]+/y);
          raw.split(/(\n[ \t]*\n(?:[ \t]*\n)*)/).forEach((part, index) => {
            if (index % 2) { if (inlineOnly) inline(' '); else paragraph = null; }
            else if (part) inline(part.replace(/\s+/g, ' ').replaceAll('~', '\u00a0').replaceAll('---', '—').replaceAll('--', '–').replaceAll("``", '“').replaceAll("''", '”'));
          });
          continue;
        }
        c.i++;
        const cmd = c.take(/[A-Za-z@]+/y) || source[c.i++] || '';
        const starred = source[c.i] === '*' && /^[a-zA-Z]+$/.test(cmd);
        if (starred) c.i++;
        if (cmd === 'begin') {
          const name = c.group() || '';
          const body = c.environment(name);
          if (mathEnvs.test(name)) {
            const inner = name.replace('*', '');
            const wrap = inner === 'align' || inner === 'flalign' ? 'aligned' : inner === 'gather' || inner === 'multline' ? 'gathered' : inner === 'alignat' ? 'alignedat' : '';
            const tex = body.replace(/\\label\s*\{[^{}]*\}/g, '');
            const el = formula(wrap ? '\\begin{' + wrap + '}' + tex + '\\end{' + wrap + '}' : tex, inner !== 'math');
            if (inner === 'math') inline(el); else block(el);
          } else if (['itemize','enumerate','description'].includes(name)) {
            const el = make(name === 'enumerate' ? 'ol' : 'ul'); block(el); flow(body, el, depth + 1, false, true);
          } else if (name === 'tabular') {
            block(table(body));
          } else if (/^(verbatim\*?|lstlisting|minted)$/.test(name)) {
            const el = make('pre', 'tex-verbatim'); el.textContent = body.replace(/^\n/, ''); block(el);
          } else {
            const supported = ['document','abstract','quote','quotation','center','flushleft','flushright','figure','figure*','table','table*','theorem','proof','definition'];
            if (!supported.includes(name)) warn('环境 ' + name + ' 的内容按正文显示');
            const el = make(name.startsWith('figure') ? 'figure' : /^(quote|quotation)$/.test(name) ? 'blockquote' : 'div');
            block(el);
            if (name === 'abstract') { const h = make('h3'); h.textContent = 'Abstract'; el.append(h); }
            flow(body.replace(/^\s*\[[\w!]+\]/, ''), el, depth + 1);
          }
        } else if (cmd === 'DeclareMathOperator') {
          const name = c.group();
          const label = c.group();
          if (/^\\[A-Za-z]+$/.test(name || '') && label !== null && label.length < 250) {
            macros[name] = '\\operatorname' + (starred ? '*' : '') + '{' + label + '}';
          }
        } else if (cmd === 'newcommand' || cmd === 'renewcommand' || cmd === 'providecommand') {
          defineMacro(cmd);
        } else if (['title','author','date'].includes(cmd)) {
          c.group(true); meta.set(cmd, c.group() || '');
        } else if (cmd === 'maketitle') {
          for (const key of ['title','author','date']) if (meta.get(key)) {
            const el = make(key === 'title' ? 'h1' : 'p', 'tex-' + key); block(el); flow(meta.get(key), el, depth + 1, true);
          }
        } else if (['chapter','section','subsection','subsubsection','paragraph'].includes(cmd)) {
          c.group(true);
          const level = cmd === 'subsection' ? 1 : cmd === 'subsubsection' || cmd === 'paragraph' ? 2 : 0;
          const el = make('h' + (level + 2));
          if (!starred && cmd !== 'paragraph') {
            scopes[level]++; scopes.fill(0, level + 1);
            el.append(scopes.slice(0, level + 1).join('.') + '　');
          }
          block(el); flow(c.group() || '', el, depth + 1, true);
        } else if (cmd === 'item' && list) {
          const li = make('li'); host.append(li); current = li; paragraph = null;
          const label = c.group(true); if (label) flow(label + ' ', li, depth + 1, true);
        } else if (Object.hasOwn(formatting, cmd)) {
          const el = make(formatting[cmd]); inline(el); flow(c.group() || '', el, depth + 1, true);
        } else if (cmd === 'href') {
          c.group(); const el = make('span'); inline(el); flow(c.group() || '', el, depth + 1, true);
        } else if (cmd === 'url' || cmd === 'path') {
          const el = make('code'); el.textContent = c.group() || ''; inline(el);
        } else if (cmd === 'caption') {
          c.group(true); const el = make('p', 'tex-caption'); block(el); flow(c.group() || '', el, depth + 1, true);
        } else if (cmd === 'verb') {
          const delimiter = source[c.i++];
          const end = delimiter ? source.indexOf(delimiter,c.i) : -1;
          const el = make('code'); el.textContent = source.slice(c.i,end < 0 ? source.length : end); inline(el);
          c.i = end < 0 ? source.length : end + 1;
        } else if (textMacros.has('\\' + cmd)) {
          const {body,arity} = textMacros.get('\\' + cmd);
          const args = Array.from({length: Math.min(9, arity)}, () => c.group() || '');
          const size = [...body.matchAll(/#([1-9])/g)].reduce(
            (length, match) => length + (args[Number(match[1])-1] || '').length, body.length);
          expandedTextLength += size;
          if (expandedTextLength > 150000) { warn('较长或递归的正文宏已折叠'); inline('\\' + cmd); continue; }
          const el = make('span'); inline(el);
          flow(body.replace(/#([1-9])/g, (_, n) => args[Number(n)-1] || ''), el, depth + 1, true);
        } else if (cmd === 'par' || cmd === 'newpage' || cmd === 'clearpage') { paragraph = null; }
        else if (cmd === '\\' || cmd === 'newline' || cmd === 'linebreak') { inline(make('br')); c.group(true); }
        else if (Object.hasOwn(symbols, cmd)) inline(symbols[cmd]);
        else if (cmd === 'today') inline(new Date().toLocaleDateString());
        else if (/^[#$%&{}_]$/.test(cmd)) inline(cmd);
        else if (/^[ ,;:!]$/.test(cmd)) inline(cmd === '!' ? '' : ' ');
        else if (['documentclass','usepackage','RequirePackage','label','pagestyle','thispagestyle'].includes(cmd)) { c.group(true); c.group(); }
        else if (!['centering','noindent','smallskip','medskip','bigskip','normalfont'].includes(cmd)) {
          const raw = c.i; c.group(true); c.group();
          unsupported('\\' + cmd, '\\' + cmd + (starred ? '*' : '') + source.slice(raw,c.i));
        }
      }
    }

    const cleaned = stripComments(String(source).replace(/\r\n?/g, '\n'));
    const start = /\\begin\s*\{document\}/.exec(cleaned);
    if (start) {
      // Preamble declarations are evaluated into local preview state. Layout commands stay off the page.
      flow(cleaned.slice(0,start.index), document.createDocumentFragment());
      warnings.clear();
      const c = new Cursor(cleaned, warn);
      c.i = start.index + start[0].length;
      flow(c.environment('document'), fragment);
    } else {
      flow(cleaned, fragment);
    }
    if (truncated) warn('文档较长，快速预览仅显示开头部分');
    target.replaceChildren(fragment);
    return { warnings: Array.from(warnings), mathCount };
  }

  window.LatexPreview = { render };
})();
