const byId = (id) => document.getElementById(id);
const source = byId('source');
const preview = byId('preview');
const storageKey = 'tools/latex-editor/draft-v1';
const defaultName = 'geometric-series.tex';
const example = String.raw`\documentclass{article}
\usepackage{amsmath}

\title{The Geometric Series}
\author{Mathematical Notes}
\date{}

\begin{document}
\maketitle

\section{A finite sum}
Start with $n$ powers of $r$. Multiplying the sum by $r$ shifts every term one place. Subtract the two sums and only the endpoints remain:
\[
  (1-r)\sum_{k=0}^{n-1} r^k = 1-r^n.
\]

\section{Taking the limit}
When $|r|<1$, the last power tends to zero. The infinite series has the value
\[
  \sum_{k=0}^{\infty} r^k = \frac{1}{1-r}.
\]
Try $r=1/2$. The partial sums approach $2$.
\end{document}
`;
let name = defaultName;
let baseline = example;
let modified = false;
let newline = '\n';
let bom = false;
let importVersion = 0;
let editVersion = 0;
let previewTimer;
let draftTimer;
let activeCompile;
let pdfUrl;

function setName(value) {
  name = value;
  byId('filename').textContent = value;
  byId('filename').title = value;
}

function persistDraft() {
  clearTimeout(draftTimer);
  try {
    localStorage.setItem(storageKey, JSON.stringify({name, source: source.value, newline, bom, modified}));
    byId('draft-status').textContent = '草稿已保存在此浏览器';
  } catch {
    byId('draft-status').textContent = '浏览器未能保存草稿，请下载源码';
  }
}

function restoreDraft() {
  source.value = example;
  try {
    const draft = JSON.parse(localStorage.getItem(storageKey));
    if (!draft || typeof draft.source !== 'string' || typeof draft.name !== 'string') return;
    source.value = draft.source;
    baseline = draft.modified ? null : draft.source;
    modified = draft.modified === true;
    newline = ['\r\n', '\r'].includes(draft.newline) ? draft.newline : '\n';
    bom = draft.bom === true;
    setName(draft.name);
    byId('draft-status').textContent = '已恢复本机草稿';
  } catch { /* The example is already open if browser storage is unavailable. */ }
}

function showPreview() {
  clearTimeout(previewTimer);
  const value = source.value;
  const diagnostics = byId('diagnostics');
  const list = byId('warning-list');
  list.replaceChildren();
  diagnostics.hidden = true;
  if (value.length > 250000) {
    preview.textContent = '文档较长，可使用下方的 PDF 编译查看完整排版。';
    byId('preview-status').textContent = '即时预览已暂停';
    return;
  }
  try {
    const result = window.LatexPreview.render(value, preview);
    if (!value.trim()) preview.textContent = '在源码栏输入内容即可预览。';
    const warnings = [...new Set(result.warnings)];
    for (const warning of warnings) {
      const item = document.createElement('li');
      item.textContent = warning;
      list.append(item);
    }
    diagnostics.hidden = warnings.length === 0;
    byId('diagnostics-title').textContent = `预览提示（${warnings.length}）`;
    byId('preview-status').textContent = warnings.length ? `${warnings.length} 项预览提示 ↓` : '已同步';
  } catch {
    preview.textContent = '无法解析当前源码。可以继续编辑，或用 PDF 编译查看日志。';
    byId('preview-status').textContent = '预览未完成';
  }
}

function showPosition() {
  const left = source.value.slice(0, source.selectionStart);
  const line = left.split('\n').length;
  const column = left.length - left.lastIndexOf('\n');
  byId('position').textContent = `行 ${line} · 列 ${column}`;
}

function setView(view) {
  byId('workspace').dataset.view = view;
  for (const button of document.querySelectorAll('[data-view][type="button"]')) {
    button.setAttribute('aria-pressed', String(button.dataset.view === view));
  }
}

function resetPdf() {
  if (activeCompile) activeCompile.abort();
  byId('pdf-viewer').hidden = true;
  byId('pdf-viewer').removeAttribute('src');
  byId('pdf-download').hidden = true;
  byId('pdf-download').removeAttribute('href');
  if (pdfUrl) URL.revokeObjectURL(pdfUrl);
  pdfUrl = null;
  byId('pdf-status').hidden = true;
  byId('pdf-log').hidden = true;
}

function onEdit() {
  editVersion++;
  modified = source.value !== baseline;
  resetPdf();
  byId('file-status').textContent = '';
  byId('draft-status').textContent = '正在保存草稿…';
  byId('preview-status').textContent = '正在排版…';
  clearTimeout(previewTimer);
  clearTimeout(draftTimer);
  previewTimer = setTimeout(showPreview, 180);
  draftTimer = setTimeout(persistDraft, 400);
  showPosition();
}

async function openFile(file) {
  const version = ++importVersion;
  const editAtStart = editVersion;
  if (!file) return;
  if (!/\.tex$/i.test(file.name)) {
    byId('file-status').textContent = '请选择 .tex 源文件。';
    return;
  }
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    const text = new TextDecoder('utf-8', {fatal: true}).decode(bytes);
    if (version !== importVersion || editAtStart !== editVersion) return;
    if (modified && !window.confirm('打开这个文件会替换当前草稿。继续打开？')) return;
    resetPdf();
    byId('attachments').value = '';
    source.value = text;
    baseline = source.value;
    modified = false;
    newline = text.match(/\r\n|\r|\n/)?.[0] || '\n';
    bom = bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf;
    setName(file.name);
    setView('split');
    source.setSelectionRange(0, 0);
    source.scrollTop = 0;
    preview.parentElement.scrollTop = 0;
    showPosition();
    showPreview();
    persistDraft();
    byId('file-status').textContent = '已打开本地文件';
  } catch (error) {
    if (version === importVersion) byId('file-status').textContent = error instanceof TypeError
      ? '文件需要使用 UTF-8 编码。' : '读取失败，请重新选择文件。';
  }
}

function downloadSource() {
  const text = (bom ? '\uFEFF' : '') + source.value.replace(/\n/g, newline);
  const url = URL.createObjectURL(new Blob([text], {type: 'application/x-tex;charset=utf-8'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
  modified = false;
  baseline = source.value;
  persistDraft();
  byId('file-status').textContent = '已导出源码副本';
}

function compilationStatus(message) {
  byId('pdf-status').hidden = false;
  byId('pdf-status').textContent = message;
}

async function compile() {
  if (activeCompile) return;
  resetPdf();
  const controller = new AbortController();
  activeCompile = controller;
  byId('compile').disabled = true;
  byId('cancel').hidden = false;
  compilationStatus('正在加载 PDF 引擎…');
  const current = source.value;
  const files = [...byId('attachments').files];
  try {
    const {compilePDF} = await import('./engine.js');
    controller.signal.throwIfAborted();
    const {blob, log} = await compilePDF(current, {signal: controller.signal, files,
      onProgress: (message) => { if (!controller.signal.aborted) compilationStatus(message); }});
    controller.signal.throwIfAborted();
    pdfUrl = URL.createObjectURL(blob);
    const viewer = byId('pdf-viewer');
    viewer.src = pdfUrl;
    viewer.hidden = false;
    const link = byId('pdf-download');
    link.href = pdfUrl;
    link.download = name.replace(/\.tex$/i, '') + '.pdf';
    link.hidden = false;
    byId('log-content').textContent = log;
    byId('pdf-log').open = false;
    byId('pdf-log').hidden = !log;
    compilationStatus('PDF 已生成');
  } catch (error) {
    if (controller.signal.aborted) return;
    const missing = /File\s+[`']([^`'\n]+)'\s+not found/i.exec(error.log || '');
    compilationStatus(missing ? `找不到 ${missing[1]}。若文件与主文档在同一目录，请添加附件后重新编译。` : error.message || '编译失败');
    byId('log-content').textContent = error.log || '';
    byId('pdf-log').hidden = !error.log;
    byId('pdf-log').open = Boolean(error.log);
  } finally {
    activeCompile = null;
    byId('compile').disabled = false;
    byId('cancel').hidden = true;
  }
}

byId('open').addEventListener('click', () => byId('file').click());
byId('file').addEventListener('change', (event) => {
  const file = event.target.files[0];
  event.target.value = '';
  openFile(file);
});
byId('download').addEventListener('click', downloadSource);
byId('compile').addEventListener('click', compile);
byId('cancel').addEventListener('click', () => { activeCompile?.abort(); compilationStatus('已取消编译'); });
byId('attachments').addEventListener('change', resetPdf);
source.addEventListener('input', onEdit);
source.addEventListener('selectionchange', showPosition);
for (const button of document.querySelectorAll('[data-view][type="button"]')) {
  button.addEventListener('click', () => setView(button.dataset.view));
}
document.addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && !event.altKey && event.key.toLowerCase() === 's') {
    event.preventDefault();
    downloadSource();
  }
});
window.addEventListener('pagehide', () => { if (modified || name !== defaultName) persistDraft(); });
restoreDraft();
source.setSelectionRange(0, 0);
showPosition();
showPreview();
