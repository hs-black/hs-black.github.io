const byId = id => document.getElementById(id);

export function createEditor({
  storageKey, defaultName, example, acceptsFile, fileHint, mime, render,
  onSourceChange, onOpen, previewLimit = 250000,
  previewLimitMessage = '文档较长。仍可编辑和下载完整源码。',
}) {
  const source = byId('source');
  const preview = byId('preview');
  let name = defaultName;
  let baseline = example;
  let modified = false;
  let newline = '\n';
  let bom = false;
  let importVersion = 0;
  let editVersion = 0;
  let previewTimer;
  let draftTimer;
  let openedFile = false;

  function setName(value) {
    name = value;
    byId('filename').textContent = value;
    byId('filename').title = value;
  }

  function persistDraft() {
    clearTimeout(draftTimer);
    draftTimer = undefined;
    try {
      localStorage.setItem(storageKey, JSON.stringify({name, source: source.value, newline, bom, modified, openedFile}));
      byId('draft-status').textContent = '草稿已保存在此浏览器';
    } catch {
      byId('draft-status').textContent = '浏览器未能保存草稿，请下载源码';
    }
  }

  function restoreDraft() {
    source.value = example;
    setName(defaultName);
    try {
      const draft = JSON.parse(localStorage.getItem(storageKey));
      if (!draft || typeof draft.source !== 'string' || typeof draft.name !== 'string') return;
      source.value = draft.source;
      baseline = draft.modified ? null : draft.source;
      modified = draft.modified === true;
      newline = ['\r\n', '\r'].includes(draft.newline) ? draft.newline : '\n';
      bom = draft.bom === true;
      setName(draft.name);
      openedFile = draft.openedFile ?? (draft.name !== defaultName);
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
    if (value.length > previewLimit) {
      preview.textContent = previewLimitMessage;
      byId('preview-status').textContent = '即时预览已暂停';
      return;
    }
    try {
      const result = render(value, preview);
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
      preview.textContent = '无法解析当前源码。请检查语法后再预览。';
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

  function onEdit() {
    editVersion++;
    modified = source.value !== baseline;
    onSourceChange?.();
    byId('file-status').textContent = '';
    byId('draft-status').textContent = '正在保存草稿…';
    byId('preview-status').textContent = '正在排版…';
    clearTimeout(previewTimer);
    clearTimeout(draftTimer);
    previewTimer = setTimeout(showPreview, 180);
    draftTimer = setTimeout(persistDraft, 400);
    showPosition();
  }

  function switchDocument(text, filename, {fromDisk, hasBom = false}) {
    importVersion++;
    editVersion++;
    source.value = text;
    baseline = source.value;
    modified = false;
    openedFile = fromDisk;
    newline = text.match(/\r\n|\r|\n/)?.[0] || '\n';
    bom = hasBom;
    setName(filename);
    onOpen?.();
    onSourceChange?.();
    setView('split');
    source.setSelectionRange(0, 0);
    source.scrollTop = 0;
    preview.parentElement.scrollTop = 0;
    showPosition();
    showPreview();
    persistDraft();
    byId('file-status').textContent = fromDisk ? '已打开本地文件' : '';
  }

  function replaceSource(text, filename) {
    if ((modified || openedFile) && !window.confirm('新建文档会替换当前工作区里的内容。继续新建？')) return false;
    switchDocument(text, filename, {fromDisk: false});
    source.focus();
    return true;
  }

  async function openFile(file) {
    const version = ++importVersion;
    const editAtStart = editVersion;
    if (!file) return;
    if (!acceptsFile(file.name)) {
      byId('file-status').textContent = fileHint;
      return;
    }
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      const text = new TextDecoder('utf-8', {fatal: true}).decode(bytes);
      if (version !== importVersion || editAtStart !== editVersion) return;
      if (modified && !window.confirm('打开这个文件会替换当前草稿。继续打开？')) return;
      const hasBom = bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf;
      switchDocument(text, file.name, {fromDisk: true, hasBom});
    } catch (error) {
      if (version === importVersion) byId('file-status').textContent = error instanceof TypeError
        ? '文件需要使用 UTF-8 编码。' : '读取失败，请重新选择文件。';
    }
  }

  function downloadSource() {
    const text = (bom ? '\uFEFF' : '') + source.value.replace(/\n/g, newline);
    const url = URL.createObjectURL(new Blob([text], {type: mime}));
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

  byId('open').addEventListener('click', () => byId('file').click());
  byId('file').addEventListener('change', (event) => {
    const file = event.target.files[0];
    event.target.value = '';
    openFile(file);
  });
  byId('download').addEventListener('click', downloadSource);
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
  window.addEventListener('pagehide', () => { if (draftTimer !== undefined) persistDraft(); });
  restoreDraft();
  source.setSelectionRange(0, 0);
  showPosition();
  showPreview();

  return {refreshPreview: showPreview, replaceSource, getFilename: () => name};
}
