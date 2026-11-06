#!/usr/bin/env python3
"""Check encrypted publishing and reading using synthetic temporary fixtures only.

Requires Node.js >= 20, Python Playwright, and Chromium. Run from any directory:
    python3 scripts/check-encrypted-post.py

No existing post, draft, or password is read or modified. Plaintext fixtures and
all generated pages live in a temporary directory outside the repository.
"""
from base64 import b64decode, b64encode
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from threading import Thread
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'scripts' / 'encrypt-post.mjs'
PASSWORD = '  synthetic fixture 密码  '
SENTINEL = 'ENCRYPTED_FIXTURE_7c23f5'
TITLE = 'Synthetic <svg onload="window.titleInjection=true"> & "title"'
FRAGMENT = r'''
<h2>ENCRYPTED_FIXTURE_7c23f5</h2>
<p>Only the synthetic reader fixture contains this sentence.</p>
<section data-math-scope="scoped-test"><p id="scoped-formula">
\(\htmlData{symbol=fixture-x}{x}+\sqrt{2}=\frac{3}{2}\)
</p></section>
<script type="application/json" id="math-definitions">
{"global":{"fixture-x":{"label":"Global fixture","description":"Global definition."}},
"scopes":{"scoped-test":{"fixture-x":{"label":"Scoped fixture","description":"Definition local to this fixture."}}}}
</script>
<figure class="article-figure" id="synthetic-step"><h3>Two states</h3>
<p id="step-value"></p><figcaption>A synthetic interactive figure.</figcaption></figure>
<button id="grow" type="button">Expand fixture</button><div id="extra"></div>
<p><a id="synthetic-reference" href="REFERENCE_URL">Open synthetic reference</a></p>
<script>
ArticleSteps.mount('#synthetic-step', {
  steps: [{title:'Initial',caption:'Zero'}, {title:'Changed',caption:'One'}],
  render(step,index) { document.getElementById('step-value').textContent=String(index); }
});
document.getElementById('grow').onclick = () => {
  document.getElementById('extra').innerHTML='<p>Additional synthetic paragraph.</p>'.repeat(40);
};
window.parentReadable=false;
try { parent.document.body; window.parentReadable=true; } catch (_) {}
fetch('https://example.invalid/encrypted-fixture-probe').catch(() => {});
</script>
'''


def run_cli(source, output, *, password=PASSWORD, expected=0, extra=()):
    command = ['node', str(CLI), '--source', str(source), '--slug', 'synthetic-check',
               '--title', TITLE, '--date', '2026-01-01', '--output', str(output),
               '--math', '--steps', '--password-stdin', *extra]
    result = subprocess.run(command, input=password + '\n', capture_output=True,
                            text=True, cwd=ROOT, timeout=30)
    assert result.returncode == expected, result.stderr
    assert PASSWORD not in result.stdout + result.stderr
    assert SENTINEL not in result.stdout + result.stderr
    return result


def read_payload(page):
    match = re.search(r'<script\b[^>]*id="encrypted-post-payload"[^>]*>(.*?)</script>', page, re.S)
    assert match, 'Missing encrypted envelope'
    return json.loads(match.group(1))


def with_payload(page, data):
    return re.sub(r'(<script\b[^>]*id="encrypted-post-payload"[^>]*>).*?(</script>)',
                  lambda match: match[1] + json.dumps(data) + match[2], page, flags=re.S)


def prepare_fixtures(folder, reference_url):
    source = folder / 'source.html'
    fragment = FRAGMENT.replace('REFERENCE_URL', reference_url)
    source.write_text(fragment, encoding='utf-8')
    first, second = folder / 'first', folder / 'second'
    run_cli(source, first)
    run_cli(source, second)
    pages = [p.joinpath('index.html').read_text() for p in (first, second)]
    envelopes = [read_payload(page) for page in pages]
    for directory, page, envelope in zip((first, second), pages, envelopes):
        assert [p.name for p in directory.iterdir()] == ['index.html']
        assert SENTINEL not in page and PASSWORD not in page
        assert 'Scoped fixture' not in page and 'synthetic-step' not in page
        assert envelope['version'] == 1
        assert envelope['cipher'] == 'AES-GCM' and envelope['kdf'] == 'PBKDF2-SHA256'
        assert envelope['iterations'] == 600000
        assert len(b64decode(envelope['salt'])) == 16
        assert len(b64decode(envelope['iv'])) == 12
        assert len(b64decode(envelope['ciphertext'])) > 16
    for key in ('salt', 'iv', 'ciphertext'):
        assert envelopes[0][key] != envelopes[1][key], f'{key} was reused'

    # Replacing an existing encrypted page is explicit and keeps only ciphertext.
    before = (first / 'index.html').read_bytes()
    run_cli(source, first, expected=1)
    assert (first / 'index.html').read_bytes() == before
    run_cli(source, first, extra=('--replace',))
    assert (first / 'index.html').read_bytes() != before
    page = (first / 'index.html').read_text()

    # Rejections operate only on temporary outputs and an existing public template.
    blocked_output = folder / 'blocked'
    run_cli(ROOT / 'templates/article.html', blocked_output, expected=1)
    assert not blocked_output.exists()
    link = folder / 'linked-source.html'
    link.symlink_to(ROOT / 'templates/article.html')
    run_cli(link, blocked_output, expected=1)
    assert not blocked_output.exists()
    ordinary = folder / 'ordinary'
    ordinary.mkdir()
    ordinary_page = ordinary / 'index.html'
    ordinary_page.write_text('<p>A synthetic ordinary page.</p>')
    original = ordinary_page.read_bytes()
    run_cli(source, ordinary, expected=1, extra=('--replace',))
    assert ordinary_page.read_bytes() == original
    run_cli(source, blocked_output, password='short', expected=1)
    assert not blocked_output.exists()
    assert source.read_text() == fragment

    corrupt = read_payload(page)
    ciphertext = bytearray(b64decode(corrupt['ciphertext']))
    ciphertext[-1] ^= 1  # Authentication tag corruption must fail before display.
    corrupt['ciphertext'] = b64encode(ciphertext).decode()
    malformed = read_payload(page)
    malformed['iterations'] = 1
    long_source = folder / 'long-source.html'
    long_source.write_text('<p><a id="long-jump" href="#long-end">Jump to document end</a></p>'
                           + fragment + '<div id="long-fixture">'
                           + '<p>Long synthetic paragraph.</p>' * 1000
                           + '</div><p id="long-end">Synthetic document end.</p>')
    long_output = folder / 'long-output'
    run_cli(long_source, long_output)
    print('PASS encrypted-only output, fresh salt/IV, explicit replacement, source boundaries and ordinary-page protection')
    return {'/fixture/': page, '/tampered/': with_payload(page, corrupt),
            '/malformed/': with_payload(page, malformed),
            '/long/': (long_output / 'index.html').read_text()}


def check_browser(base):
    with sync_playwright() as playwright:
        executable = (shutil.which(os.environ.get('CHROMIUM_PATH', 'chromium'))
                      or shutil.which('chromium-browser') or playwright.chromium.executable_path)
        browser = playwright.chromium.launch(executable_path=executable)
        context = browser.new_context(viewport={'width': 1280, 'height': 900}, reduced_motion='reduce')
        page = context.new_page()
        errors, requests = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: requests.append((request.url, request.post_data or '')))
        page.add_init_script('''window.__storageWrites=0;
          const original=Storage.prototype.setItem;
          Storage.prototype.setItem=function(...args){window.__storageWrites++;return original.apply(this,args);};''')

        def visit(path='/fixture/'):
            page.goto(base + path, wait_until='networkidle')

        def submit(value=PASSWORD):
            page.locator('#encrypted-post-password').fill(value)
            page.locator('#encrypted-post-form').evaluate('(form)=>form.requestSubmit()')

        def unlock():
            submit()
            page.locator('iframe.encrypted-post-frame').wait_for()
            frame = page.frame_locator('iframe.encrypted-post-frame')
            frame.get_by_text(SENTINEL, exact=True).wait_for()
            return frame

        def locked():
            assert page.locator('iframe').count() == 0
            assert page.locator('#encrypted-post-gate').is_visible()
            assert page.locator('#encrypted-post-password').input_value() == ''
            assert SENTINEL not in page.content()

        visit()
        locked()
        assert page.locator('h1').inner_text() == TITLE
        assert not page.locator('h1 svg').count()
        assert page.evaluate('window.titleInjection === undefined')
        submit(PASSWORD.strip())  # Leading/trailing spaces and Unicode are significant.
        page.wait_for_function("document.querySelector('#encrypted-post-status').dataset.state==='error'")
        locked()
        frame = unlock()
        assert page.locator('#encrypted-post-password').input_value() == ''
        permissions = page.locator('iframe').get_attribute('sandbox').split()
        assert set(permissions) == {'allow-scripts', 'allow-popups', 'allow-popups-to-escape-sandbox'}
        assert frame.locator('body').evaluate('()=>window.parentReadable') is False
        frame.locator('#scoped-formula .katex').wait_for()
        assert not frame.locator('.katex-error').count()
        loaded = frame.locator('body').evaluate('''async()=>{
          const faces=await document.fonts.load('16px KaTeX_Main','M');
          await document.fonts.ready;
          return faces.some(face=>face.status==='loaded') && document.fonts.check('16px KaTeX_Main','M');
        }''')
        assert loaded, 'KaTeX must use successfully loaded bundled fonts in the opaque iframe'
        symbol = frame.locator('.katex-html [data-symbol="fixture-x"]').first
        symbol.hover()
        tooltip = frame.locator('#reader-symbol-tip')
        assert tooltip.is_visible() and 'Scoped fixture' in tooltip.inner_text()
        assert 'Global definition' not in tooltip.inner_text()
        page.mouse.move(0, 0)
        assert frame.locator('#step-value').inner_text() == '0'
        frame.locator('[data-action="next"]').click()
        assert frame.locator('#step-value').inner_text() == '1'
        frame.locator('[data-action="reset"]').click()
        assert frame.locator('#step-value').inner_text() == '0'

        with context.expect_page() as popup_event:
            frame.locator('#synthetic-reference').click()
        popup = popup_event.value
        popup.wait_for_load_state('domcontentloaded')
        assert urlsplit(popup.url).hostname == 'localhost'
        assert popup.locator('h1').inner_text() == 'Synthetic external reference'
        assert popup.evaluate('window.opener === null && document.referrer === ""')
        assert SENTINEL not in popup.content() and PASSWORD not in popup.content()
        assert frame.locator('body').evaluate('()=>window.parentReadable') is False
        popup.close()

        page.wait_for_function('document.querySelector("iframe").clientHeight > 300')
        height = page.locator('iframe').evaluate('(element)=>element.clientHeight')
        frame.locator('#grow').click()
        page.wait_for_function('(height)=>document.querySelector("iframe").clientHeight>height+100', arg=height)
        frame.locator('#extra').evaluate('(element)=>element.replaceChildren()')
        page.wait_for_function('(height)=>document.querySelector("iframe").clientHeight<=height+2', arg=height)
        for width, scheme in ((375, 'light'), (375, 'dark')):
            page.set_viewport_size({'width': width, 'height': 900})
            page.emulate_media(color_scheme=scheme)
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            assert frame.locator('body').evaluate('()=>document.documentElement.scrollWidth<=innerWidth')
            if scheme == 'dark':
                assert frame.locator('body').evaluate('()=>getComputedStyle(document.body).color') == 'rgb(236, 236, 241)'
        assert page.evaluate('localStorage.length===0 && sessionStorage.length===0 && __storageWrites===0')
        page.locator('#encrypted-post-lock').click()
        locked()
        unlock()
        page.reload(wait_until='networkidle')
        locked()
        print('PASS exact password, title escaping/wrapping, opaque reading, scoped math/fonts, steps, safe reference popup and mobile layout')

        # Pause key derivation deterministically, lock while it is pending, then
        # release it. A stale success must not recreate the plaintext frame.
        for action in ('pagehide', 'lock'):
            visit()
            page.evaluate('''()=>{
              const original=crypto.subtle.deriveKey.bind(crypto.subtle);
              crypto.subtle.deriveKey=async(...args)=>{
                window.__deriveStarted=true;
                await new Promise(resolve=>window.__releaseDerivation=resolve);
                try{return await original(...args);}finally{window.__deriveFinished=true;}
              };
            }''')
            submit()
            page.wait_for_function('window.__deriveStarted===true')
            if action == 'pagehide':
                page.evaluate("dispatchEvent(new PageTransitionEvent('pagehide'))")
            else:
                page.locator('#encrypted-post-lock').evaluate('(button)=>button.click()')
            locked()
            page.evaluate('window.__releaseDerivation()')
            page.wait_for_function('window.__deriveFinished===true')
            locked()

        # A very tall iframe, mobile reflow and deep iframe screenshots can
        # leave Chromium's compositor hit testing a frame behind the DOM.
        # Settle the actual reading viewport before clicking the visible lock.
        visit('/long/')
        page.set_viewport_size({'width': 1280, 'height': 900})
        long_frame = unlock()
        page.wait_for_function('document.querySelector("iframe").clientHeight>40000')
        page.set_viewport_size({'width': 375, 'height': 900})
        long_frame.locator('#long-jump').click()
        page.wait_for_function('scrollY>35000')
        assert long_frame.locator('body').evaluate('()=>location.href') == 'about:srcdoc'
        assert long_frame.locator('iframe').count() == 0
        assert long_frame.get_by_text(SENTINEL, exact=True).count() == 1
        long_frame.locator('#long-end').scroll_into_view_if_needed()
        long_frame.locator('#long-end').screenshot()
        page.evaluate('scrollTo(0, 0)')
        page.wait_for_function('scrollY===0')
        page.wait_for_timeout(100)
        page.locator('#encrypted-post-lock').click()
        locked()

        visit('/tampered/')
        submit()
        page.wait_for_function("document.querySelector('#encrypted-post-status').dataset.state==='error'")
        locked()
        visit('/malformed/')
        assert page.locator('#encrypted-post-submit').is_disabled()
        locked()
        assert not errors, errors
        assert all(urlsplit(url).hostname in ('127.0.0.1', 'localhost') for url, _ in requests), requests
        allowed_paths = {'/fixture/', '/tampered/', '/malformed/', '/long/', '/assets/site.css',
                         '/assets/encrypted-post.css', '/assets/encrypted-post.js',
                         '/assets/favicon.svg'}
        for url, body in requests:
            parsed = urlsplit(url)
            assert parsed.path in allowed_paths and not body, (url, body)
            assert not parsed.query or re.fullmatch(r'v=[a-f0-9]{12}', parsed.query), url
            assert PASSWORD not in url + body and SENTINEL not in url + body
        assert page.evaluate('localStorage.length===0 && sessionStorage.length===0 && __storageWrites===0')
        print('PASS tamper rejection, malformed envelope, pending-unlock cancellation, no persisted secrets or outbound requests')
        context.close()
        browser.close()


def main():
    with TemporaryDirectory(prefix='encrypted-post-check-') as temporary:
        fixtures = {}
        reference_requests = []

        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                path = urlsplit(self.path).path
                if path in fixtures:
                    body = fixtures[path].encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == '/reference/':
                    reference_requests.append(dict(self.headers))
                    body = b'<!doctype html><html><h1>Synthetic external reference</h1></html>'
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(body)
                elif path.startswith('/assets/'):
                    super().do_GET()
                else:
                    self.send_error(404)

        server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(ROOT)))
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            reference_url = f'http://localhost:{server.server_port}/reference/'
            fixtures.update(prepare_fixtures(Path(temporary), reference_url))
            check_browser(f'http://127.0.0.1:{server.server_port}')
            assert len(reference_requests) == 1
            assert all(key.lower() != 'referer' for key in reference_requests[0])
            assert PASSWORD not in str(reference_requests) and SENTINEL not in str(reference_requests)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    print('Encrypted-post checks passed; temporary fixtures removed.')


if __name__ == '__main__':
    main()
