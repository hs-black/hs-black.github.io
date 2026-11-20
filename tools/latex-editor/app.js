import {createEditor} from '../editor-workspace.js';

const byId = id => document.getElementById(id);
const source = byId('source');
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

let activeCompile;
let pdfUrl;

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
    link.download = workspace.getFilename().replace(/\.tex$/i, '') + '.pdf';
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

const workspace = createEditor({
  storageKey: 'tools/latex-editor/draft-v1',
  defaultName: 'geometric-series.tex',
  example,
  acceptsFile: name => /\.tex$/i.test(name),
  fileHint: '请选择 .tex 源文件。',
  mime: 'application/x-tex;charset=utf-8',
  render: (text, target) => window.LatexPreview.render(text, target),
  previewLimitMessage: '文档较长，可使用下方的 PDF 编译查看完整排版。',
  onSourceChange: resetPdf,
  onOpen: () => { byId('attachments').value = ''; },
});

byId('compile').addEventListener('click', compile);
byId('cancel').addEventListener('click', () => { activeCompile?.abort(); compilationStatus('已取消编译'); });
byId('attachments').addEventListener('change', resetPdf);
