#!/usr/bin/env python3
"""Check published articles, math and inline figures in a local browser.

Requires Python Playwright and Chromium. Run `python3 scripts/check-articles.py`.
Optional positional slugs restrict the check, e.g. `kv-cache paged-attention`.
CHROMIUM_PATH selects a browser. All HTTP resources must come from this checkout.
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
    totals = {'articles': 0, 'math': 0, 'figures': 0}
    for slug in slugs:
        events.clear()
        page.set_viewport_size({'width': 1280, 'height': 900})
        page.emulate_media(color_scheme='light')
        page.goto(base + '/posts/' + slug + '/', wait_until='networkidle')
        page.wait_for_load_state('load')
        result = page.evaluate(AUDIT)
        issues = result['issues']
        for controls in page.locator('.article-step-controls').all():
            play = controls.locator('[data-action=play]')
            slider = controls.locator('input[type=range]')
            if play.get_attribute('aria-pressed') != 'false' or float(slider.input_value()) != 0:
                issues.append('figure did not start paused at initial position')
            for value in (slider.get_attribute('max'), '0'):
                slider.fill(value)
                issues.extend(page.evaluate(AUDIT)['issues'])
        # Wide tables and equations may scroll inside their own containers.
        page.locator('details').evaluate_all('(items) => items.forEach(item => { item.open = true; })')
        issues.extend(page.evaluate(AUDIT)['issues'])
        for width in (390, 320):
            page.set_viewport_size({'width': width, 'height': 844})
            page.emulate_media(color_scheme='dark')
            issues.extend(page.evaluate(AUDIT)['issues'])
        issues = sorted(set(issues + events))
        totals['articles'] += 1
        totals['math'] += result['math']
        totals['figures'] += result['figures']
        if issues:
            failures.append({'slug': slug, 'issues': issues})
            print('FAIL', slug, json.dumps(issues, ensure_ascii=False), flush=True)
        else:
            print('PASS', slug, flush=True)
    page.close()
    print(json.dumps({'totals': totals, 'failures': failures}, ensure_ascii=False, indent=2))
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
