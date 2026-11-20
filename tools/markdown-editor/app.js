import {createEditor} from '../editor-workspace.js';

const source = document.querySelector('#source');
const imagePicker = document.querySelector('#images');
const imageStatus = document.querySelector('#image-status');
const remoteImages = document.querySelector('#remote-images');
const images = new Map();
const example = [
  '# 推理延迟笔记',
  '',
  '一次请求生成了 **5 个 token**。它们是什么时候到达的？',
  '',
  '| 事件 | 请求发出后的时间 |',
  '| :--- | ---: |',
  '| 首个 token 到达 | 0.30 秒 |',
  '| 最后一个 token 到达 | 1.10 秒 |',
  '',
  '## 平均输出间隔',
  '',
  '从第一个到第五个 token，共经过 4 个间隔。',
  '',
  '$$',
  '\\Delta t = \\frac{1.10 - 0.30}{5 - 1} = 0.20\\,\\mathrm{s}',
  '$$',
  '',
  '```python',
  'gap = (1.10 - 0.30) / (5 - 1)',
  '```',
  '',
  '- [x] 记录首个 token 的时间',
  '- [ ] 再测一次，比较两次请求的平均输出间隔',
  '',
].join('\n');

function resetImages() {
  for (const url of images.values()) URL.revokeObjectURL(url);
  images.clear();
  imagePicker.value = '';
  imageStatus.textContent = '';
  remoteImages.checked = false;
}

const editor = createEditor({
  storageKey: 'tools/markdown-editor/draft-v1',
  defaultName: '推理延迟笔记.md',
  example,
  acceptsFile: name => /\.(md|markdown|mdown)$/i.test(name),
  fileHint: '请选择 .md、.markdown 或 .mdown 文件。',
  mime: 'text/markdown;charset=utf-8',
  previewLimit: 1000000,
  previewLimitMessage: '文档超过 100 万字符。可继续编辑并下载完整源码。',
  render: (value, target) => window.MarkdownPreview.render(value, target, images, {
    allowRemoteImages: remoteImages.checked,
  }),
  onOpen: resetImages,
});

document.querySelector('#new').addEventListener('click', () => editor.replaceSource('', '未命名.md'));
remoteImages.addEventListener('change', () => editor.refreshPreview());
imagePicker.addEventListener('change', () => {
  let skipped = 0;
  for (const file of imagePicker.files) {
    if (!file.type.startsWith('image/')) { skipped++; continue; }
    if (images.has(file.name)) URL.revokeObjectURL(images.get(file.name));
    images.set(file.name, URL.createObjectURL(file));
  }
  imagePicker.value = '';
  imageStatus.textContent = `已载入 ${images.size} 张图片` + (skipped ? `，跳过 ${skipped} 个非图片文件` : '');
  editor.refreshPreview();
});

function formatSelection(kind) {
  const start = source.selectionStart;
  const end = source.selectionEnd;
  const text = source.value.slice(start, end) || (kind === 'code' ? '代码' : '文字');
  let marker = kind === 'bold' ? '**' : '*';
  let padding = '';
  if (kind === 'code') {
    const runs = text.match(/`+/g) || [];
    marker = '`'.repeat(1 + runs.reduce((max, run) => Math.max(max, run.length), 0));
    if (/^`|`$/.test(text)) padding = ' ';
  }
  const formatted = marker + padding + text + padding + marker;
  source.focus();
  // insertText keeps the entire formatting action in the textarea's native undo history.
  if (!document.execCommand('insertText', false, formatted)) {
    source.setRangeText(formatted, start, end, 'end');
    source.dispatchEvent(new Event('input', {bubbles: true}));
  }
  const textStart = start + marker.length + padding.length;
  source.setSelectionRange(textStart, textStart + text.length);
}
for (const button of document.querySelectorAll('[data-format]')) {
  button.addEventListener('click', () => formatSelection(button.dataset.format));
}
source.addEventListener('keydown', event => {
  if ((!event.ctrlKey && !event.metaKey) || event.altKey || event.shiftKey || event.isComposing) return;
  const kind = {b: 'bold', i: 'italic'}[event.key.toLowerCase()];
  if (kind) { event.preventDefault(); formatSelection(kind); }
});
