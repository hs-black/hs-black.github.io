#!/usr/bin/env python3
"""Check published articles, math and inline figures in a local browser.

Requires Python Playwright and Chromium. Run `python3 scripts/check-articles.py`.
Optional positional slugs restrict the check, e.g. `kv-cache paged-attention`.
CHROMIUM_PATH selects a browser. All HTTP resources must come from this checkout.
Encrypted posts are checked as locked public shells, without a password.
Run scripts/check-encrypted-post.py for encryption/decryption behavior using
synthetic content and a test password; never supply a real article password.
"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import sys
from threading import Thread
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


AUDIT = r"""() => {
  const issues = [];
  if (!document.querySelector('article.prose')) issues.push('missing article');
  if (document.querySelector('.deck-page,.deck,.stage,.slide,.deck-ui')) issues.push('obsolete presentation markup');
  if (document.querySelector('.symbol-glossary')) issues.push('obsolete symbol glossary');
  for (const el of document.querySelectorAll('.katex-error')) issues.push('invalid rendered math: ' + (el.getAttribute('title') || el.textContent));
  if (document.body.scrollWidth > innerWidth + 1) issues.push('page overflows viewport');
  const ids = [...document.querySelectorAll('[id]')].map(e => e.id);
  if (ids.length !== new Set(ids).size) issues.push('duplicate element ids');
  const sources = new Set([...document.querySelectorAll('annotation[encoding="application/x-tex"]')].map(e => e.textContent));
  for (const el of document.querySelectorAll('[data-tex]')) {
    sources.add(el.dataset.tex);
    if (el.namespaceURI === 'http://www.w3.org/1999/xhtml' && !el.querySelector('.katex')) issues.push('unrendered explicit math: ' + el.dataset.tex);
  }
  for (const tex of sources) {
    try { katex.renderToString(tex, {throwOnError:true, strict:'ignore', trust:c => c.command === '\\htmlData'}); }
    catch (error) { issues.push('invalid TeX: ' + tex + ' — ' + error.message); }
  }
  for (const el of document.querySelectorAll('[data-symbol]')) {
    if (!el.closest('.katex-mathml') && !el.classList.contains('reader-symbol')) issues.push('missing symbol definition: ' + el.dataset.symbol);
  }
  const walker = document.createTreeWalker(document.querySelector('article') || document.body, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    if (node.parentElement?.closest('script,style,pre,code,.katex,.no-math')) continue;
    if (/\\\(|\\\[|\$\$/.test(node.nodeValue)) issues.push('unrendered formula: ' + node.nodeValue.trim().slice(0,120));
    if (/\\(?:sqrt|frac|approx|cdot|times|gamma|alpha|beta|mathbf|operatorname)\b|\b(?:sqrt|frac|mathbf|operatorname)\s*\{/.test(node.nodeValue)) issues.push('unrendered math command: ' + node.nodeValue.trim().slice(0,120));
  }
  for (const el of document.querySelectorAll('a[href^="#"]')) {
    const id = decodeURIComponent(el.hash.slice(1));
    if (id && !document.getElementById(id)) issues.push('missing section: #' + id);
  }
  return {issues:[...new Set(issues)], math:document.querySelectorAll('.katex').length,
    figures:document.querySelectorAll('figure').length, exercises:document.querySelectorAll('details').length};
}"""


ENCRYPTED_AUDIT = r"""() => {
  const issues = [];
  const require = (condition, message) => { if (!condition) issues.push(message); };
  const gate = document.getElementById('encrypted-post-gate');
  const form = document.getElementById('encrypted-post-form');
  const password = document.getElementById('encrypted-post-password');
  const open = document.getElementById('encrypted-post-open');
  const content = document.getElementById('encrypted-post-content');
  const status = document.getElementById('encrypted-post-status');
  require(!!document.querySelector('.post-header h1'), 'missing public title');
  require(!!gate && !gate.hidden, 'missing visible password gate');
  require(!!form && !!password && password.type === 'password' && !password.value, 'invalid or prefilled password form');
  require(!!document.getElementById('encrypted-post-submit'), 'missing unlock control');
  require(!!document.getElementById('encrypted-post-lock'), 'missing relock control');
  require(!!status && status.getAttribute('role') === 'status', 'missing accessible status');
  require(!!open && open.hidden, 'encrypted page did not start locked');
  require(!!content && !content.childNodes.length, 'plaintext container is not empty');
  require(!document.querySelector('article.prose,iframe'), 'decrypted article exposed before unlock');
  require(!!document.querySelector('script[src^="/assets/encrypted-post.js"]'), 'missing encrypted reader script');
  require(!!document.querySelector('link[rel="stylesheet"][href^="/assets/encrypted-post.css"]'), 'missing encrypted reader style');
  require((document.querySelector('meta[name="robots"]')?.content || '').includes('noindex'), 'missing noindex metadata');
  require(document.body.scrollWidth <= innerWidth + 1, 'encrypted shell overflows viewport');
  const ids = [...document.querySelectorAll('[id]')].map(el => el.id);
  require(ids.length === new Set(ids).size, 'duplicate element ids');
  try {
    const element = document.getElementById('encrypted-post-payload');
    require(element?.type === 'application/json', 'missing encrypted payload');
    const payload = JSON.parse(element?.textContent || '');
    require(payload.version === 1 && payload.cipher === 'AES-GCM' && payload.kdf === 'PBKDF2-SHA256' && payload.iterations === 600000, 'unsupported encryption envelope');
    const length = value => {
      if (typeof value !== 'string' || value.length > 22369624 || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(value)) throw new Error();
      return atob(value).length;
    };
    require(length(payload.salt) === 16, 'invalid encryption salt');
    require(length(payload.iv) === 12, 'invalid encryption IV');
    const size = length(payload.ciphertext);
    require(size > 16 && size <= 16777216, 'invalid encrypted document size');
  } catch (_) {
    issues.push('invalid encrypted payload encoding');
  }
  return {issues:[...new Set(issues)], math:0, figures:0, exercises:0};
}"""


def check(browser, base, slugs):
    failures = []
    page = browser.new_page(viewport={'width': 1280, 'height': 900}, reduced_motion='reduce')
    events = []
    page.on('pageerror', lambda error: events.append('JS: ' + str(error)))
    page.on('response', lambda response: events.append(f'HTTP {response.status}: {response.url}') if response.status >= 400 else None)

    def local_only(route):
        if urlsplit(route.request.url).hostname in ('127.0.0.1', 'localhost'):
            route.continue_()
        else:
            events.append('External resource: ' + route.request.url)
            route.abort()

    page.route('**/*', local_only)
    totals = {'articles': 0, 'encrypted_shells': 0, 'math': 0, 'figures': 0}
    for slug in slugs:
        events.clear()
        page.set_viewport_size({'width': 1280, 'height': 900})
        page.emulate_media(color_scheme='light')
        page.goto(base + '/posts/' + slug + '/', wait_until='networkidle')
        page.wait_for_load_state('load')
        encrypted = bool(page.locator('#encrypted-post-payload, #encrypted-post-form').count())
        audit = ENCRYPTED_AUDIT if encrypted else AUDIT
        result = page.evaluate(audit)
        issues = result['issues']
        for controls in page.locator('.article-step-controls').all():
            play = controls.locator('[data-action=play]')
            slider = controls.locator('input[type=range]')
            if play.get_attribute('aria-pressed') != 'false' or float(slider.input_value()) != 0:
                issues.append('figure did not start paused at initial position')
            for value in (slider.get_attribute('max'), '0'):
                slider.fill(value)
                issues.extend(page.evaluate(audit)['issues'])
        # Wide tables and equations may scroll inside their own containers.
        page.locator('details').evaluate_all('(items) => items.forEach(item => { item.open = true; })')
        issues.extend(page.evaluate(audit)['issues'])
        for width in (390, 320):
            page.set_viewport_size({'width': width, 'height': 844})
            page.emulate_media(color_scheme='dark')
            issues.extend(page.evaluate(audit)['issues'])
        issues = sorted(set(issues + events))
        totals['encrypted_shells' if encrypted else 'articles'] += 1
        totals['math'] += result['math']
        totals['figures'] += result['figures']
        if issues:
            failures.append({'slug': slug, 'issues': issues})
            print('FAIL', slug, json.dumps(issues, ensure_ascii=False), flush=True)
        else:
            print('PASS encrypted shell' if encrypted else 'PASS', slug, flush=True)
    page.close()
    print(json.dumps({'totals': totals, 'failures': failures}, ensure_ascii=False, indent=2))
    if totals['encrypted_shells']:
        print('Encrypted pages: locked shell checks only. Run scripts/check-encrypted-post.py for synthetic encryption/decryption tests.')
    assert not failures, 'Article checks failed'


def main():
    slugs = sys.argv[1:] or sorted(p.parent.name for p in (ROOT / 'posts').glob('*/index.html'))
    for slug in slugs:
        if not (ROOT / 'posts' / slug / 'index.html').is_file():
            raise SystemExit('Unknown post: ' + slug)
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(ROOT)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            executable = (shutil.which(os.environ.get('CHROMIUM_PATH', 'chromium'))
                          or shutil.which('chromium-browser') or playwright.chromium.executable_path)
            browser = playwright.chromium.launch(executable_path=executable)
            try:
                check(browser, f'http://127.0.0.1:{server.server_port}', slugs)
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == '__main__':
    main()
