/* Password-protected posts. The public page contains only metadata and ciphertext.
 * Protocol v1: PBKDF2-SHA256 (600000), AES-256-GCM, ciphertext followed by tag.
 * Passwords, keys and decrypted documents are never persisted by this reader.
 */
(() => {
  'use strict';
  const form = document.getElementById('encrypted-post-form');
  if (!form) return;
  const password = document.getElementById('encrypted-post-password');
  const submit = document.getElementById('encrypted-post-submit');
  const status = document.getElementById('encrypted-post-status');
  const gate = document.getElementById('encrypted-post-gate');
  const opened = document.getElementById('encrypted-post-open');
  const lockButton = document.getElementById('encrypted-post-lock');
  const content = document.getElementById('encrypted-post-content');
  const payloadNode = document.getElementById('encrypted-post-payload');
  if (![password, submit, status, gate, opened, lockButton, content, payloadNode].every(Boolean)) return;

  const MAX_BYTES = 16 * 1024 * 1024;
  const ITERATIONS = 600000;
  const AAD = new TextEncoder().encode('hs-black.github.io/encrypted-post/v1');
  const CSP = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'";
  let manifest;
  let ready = false;
  let busy = false;
  let generation = 0;
  let frame = null;
  let channel = '';

  function announce(message, error = false) {
    status.textContent = message;
    status.dataset.state = error ? 'error' : 'normal';
  }
  function controls(isBusy) {
    busy = isBusy;
    form.setAttribute('aria-busy', String(isBusy));
    password.disabled = !ready || isBusy;
    submit.disabled = !ready || isBusy;
    submit.textContent = isBusy ? '正在解锁…' : '解锁文章';
  }
  function decodeBase64(value, exactLength) {
    if (typeof value !== 'string' || !value.length || value.length % 4 !== 0 ||
        value.length > Math.ceil(MAX_BYTES / 3) * 4 || !/^[A-Za-z0-9+/]*={0,2}$/.test(value)) {
      throw new Error('Invalid encrypted post');
    }
    const raw = atob(value);
    if (btoa(raw) !== value || (exactLength !== undefined && raw.length !== exactLength)) {
      throw new Error('Invalid encrypted post');
    }
    return Uint8Array.from(raw, character => character.charCodeAt(0));
  }
  function readManifest() {
    const source = payloadNode.textContent;
    if (source.length > Math.ceil(MAX_BYTES / 3) * 4 + 4096) throw new Error('Invalid encrypted post');
    const data = JSON.parse(source);
    if (!data || data.version !== 1 || data.cipher !== 'AES-GCM' || data.kdf !== 'PBKDF2-SHA256' ||
        data.iterations !== ITERATIONS) throw new Error('Invalid encrypted post');
    const salt = decodeBase64(data.salt, 16);
    const iv = decodeBase64(data.iv, 12);
    const ciphertext = decodeBase64(data.ciphertext);
    if (ciphertext.length < 16 || ciphertext.length > MAX_BYTES) throw new Error('Invalid encrypted post');
    return {salt, iv, ciphertext};
  }
  function clearFrame() {
    if (frame) {
      frame.remove();
      frame.removeAttribute('srcdoc');
      frame = null;
    }
    content.replaceChildren();
    channel = '';
  }
  function lock({focus = true, silent = false} = {}) {
    generation += 1;
    clearFrame();
    password.value = '';
    opened.hidden = true;
    gate.hidden = false;
    controls(false);
    if (!silent) announce('文章已锁定。');
    else announce('');
    if (focus && ready) password.focus();
  }
  function readerDocument(source, token) {
    if (!/<head\b[^>]*>/i.test(source) || !/<\/body\s*>/i.test(source)) {
      throw new Error('Invalid encrypted document');
    }
    // This runs in an opaque-origin iframe. Only the article height crosses back.
    const relay = `(() => {
      const channel = ${JSON.stringify(token)};
      let queued = false, previous = -1;
      // References leave the opaque reading frame only after a user action.
      // Keep fragment links inside the article; HTTP(S) targets open without
      // an opener or referrer. Authored scripts remain trusted article code.
      function prepareReference(anchor) {
        if (!anchor) return;
        const href = anchor.getAttribute('href') || '';
        if (!href || href.startsWith('#')) return;
        let target;
        try { target = new URL(href, document.baseURI); } catch (_) { return; }
        if (target.protocol !== 'https:' && target.protocol !== 'http:') return;
        anchor.target = '_blank';
        anchor.rel = 'noopener noreferrer';
        anchor.referrerPolicy = 'no-referrer';
      }
      document.querySelectorAll('a[href]').forEach(prepareReference);
      const prepareClickedReference = event => prepareReference(event.target.closest?.('a[href]'));
      document.addEventListener('click', prepareClickedReference, true);
      document.addEventListener('auxclick', prepareClickedReference, true);
      function measure() {
        queued = false;
        const body = document.body;
        if (!body) return;
        const margins = getComputedStyle(body);
        const height = Math.ceil(Math.max(body.scrollHeight, body.offsetHeight, body.getBoundingClientRect().height)
          + (parseFloat(margins.marginTop) || 0) + (parseFloat(margins.marginBottom) || 0));
        if (height !== previous) {
          previous = height;
          parent.postMessage({type: 'encrypted-post-height', channel, height}, '*');
        }
      }
      function schedule() {
        if (!queued) { queued = true; requestAnimationFrame(measure); }
      }
      new ResizeObserver(schedule).observe(document.body);
      new MutationObserver(schedule).observe(document.body, {childList:true, subtree:true, attributes:true, characterData:true});
      addEventListener('load', schedule);
      addEventListener('resize', schedule);
      document.fonts?.ready.then(schedule);
      schedule();
    })();`;
    const policy = '<meta http-equiv="Content-Security-Policy" content="' + CSP + '">';
    return source.replace(/<head\b[^>]*>/i, match => match + policy)
      .replace(/<\/body\s*>/i, '<script>' + relay + '</script></body>');
  }
  function showDocument(source) {
    const random = crypto.getRandomValues(new Uint8Array(16));
    channel = Array.from(random, byte => byte.toString(16).padStart(2, '0')).join('');
    const documentSource = readerDocument(source, channel);
    frame = document.createElement('iframe');
    frame.className = 'encrypted-post-frame';
    frame.title = (document.querySelector('.post-header h1')?.textContent || '博文') + ' · 正文';
    frame.setAttribute('sandbox', 'allow-scripts allow-popups allow-popups-to-escape-sandbox');
    frame.referrerPolicy = 'no-referrer';
    frame.tabIndex = 0;
    frame.srcdoc = documentSource;
    content.replaceChildren(frame);
    password.value = '';
    gate.hidden = true;
    opened.hidden = false;
    announce('文章已解锁。');
    // The lock control precedes the readable frame in keyboard navigation.
    lockButton.focus({preventScroll: true});
  }

  addEventListener('message', event => {
    if (!frame || event.source !== frame.contentWindow) return;
    const data = event.data;
    if (!data || data.type !== 'encrypted-post-height' || data.channel !== channel ||
        typeof data.height !== 'number' || !Number.isFinite(data.height) || data.height < 0) return;
    frame.style.height = Math.max(160, Math.min(1000000, Math.ceil(data.height))) + 'px';
  });
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (!ready || busy) return;
    if (password.value.length === 0) {
      announce('请输入密码。', true);
      password.focus();
      return;
    }
    const attempt = ++generation;
    const passwordBytes = new TextEncoder().encode(password.value);
    password.value = '';
    controls(true);
    announce('正在解锁…');
    let plaintext;
    try {
      const baseKey = await crypto.subtle.importKey('raw', passwordBytes, 'PBKDF2', false, ['deriveKey']);
      passwordBytes.fill(0);
      if (attempt !== generation) return;
      const key = await crypto.subtle.deriveKey(
        {name:'PBKDF2', salt:manifest.salt, iterations:ITERATIONS, hash:'SHA-256'},
        baseKey, {name:'AES-GCM', length:256}, false, ['decrypt']);
      if (attempt !== generation) return;
      plaintext = await crypto.subtle.decrypt(
        {name:'AES-GCM', iv:manifest.iv, additionalData:AAD, tagLength:128}, key, manifest.ciphertext);
      if (attempt !== generation) return;
      const source = new TextDecoder('utf-8', {fatal:true}).decode(plaintext);
      showDocument(source);
    } catch (_) {
      if (attempt === generation) {
        clearFrame();
        announce('未能解锁，请检查密码后重试。', true);
      }
    } finally {
      passwordBytes.fill(0);
      if (plaintext) new Uint8Array(plaintext).fill(0);
      if (attempt === generation) {
        controls(false);
        if (!gate.hidden) password.focus();
      }
    }
  });
  lockButton.addEventListener('click', () => lock());
  addEventListener('pagehide', () => lock({focus:false, silent:true}));
  addEventListener('pageshow', event => {
    if (event.persisted) lock({focus:false, silent:true});
  });

  password.value = '';
  opened.hidden = true;
  gate.hidden = false;
  if (!globalThis.crypto?.subtle || !globalThis.isSecureContext) {
    controls(false);
    announce('请通过 HTTPS 或本机预览地址打开这篇文章。', true);
    return;
  }
  try {
    manifest = readManifest();
    ready = true;
    controls(false);
  } catch (_) {
    controls(false);
    announce('文章暂时无法打开，请稍后再试。', true);
  }
})();
