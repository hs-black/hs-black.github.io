#!/usr/bin/env python3
"""Check the local Markdown editing and preview workflow with Playwright and Chromium."""
from base64 import b64decode
from codecs import BOM_UTF8
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
from threading import Thread
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
PIXEL = b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jJRkAAAAASUVORK5CYII=')
SOURCE = r"""# 实验记录

- [x] 完成推导

| 方法 | 结果 |
| --- | --- |
| A | 2 |

行内 $x^2$ 和独立公式

$$
\sum_{i=1}^{n} i
$$

```text
$literal$ and **plain text**
```

![本地图像](images/pixel.png)
![远程图像](https://example.invalid/external.png)

<b id="raw-markup-marker">原始标签</b>
"""


def file_bytes(text):
    return BOM_UTF8 + text.replace('\n', '\r\n').encode('utf-8')


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def check_editor(origin):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=shutil.which('chromium'))
        page = browser.new_page(accept_downloads=True)
        unexpected, errors = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))

        def local_requests(route):
            url = route.request.url
            if urlsplit(url).scheme in ('http', 'https') and (
                not url.startswith(origin + '/') or route.request.method not in ('GET', 'HEAD')
                or urlsplit(url).path.endswith('/pixel.png')
            ):
                unexpected.append((route.request.method, url))
                route.abort()
            else:
                route.continue_()

        page.context.route('**/*', local_requests)
        page.goto(origin + '/tools/markdown-editor/')
        page.locator('#file').set_input_files({
            'name': '笔记.md', 'mimeType': 'text/markdown', 'buffer': file_bytes(SOURCE),
        })
        source, preview = page.locator('#source'), page.locator('#preview')
        expect(source).to_have_value(SOURCE)
        expect(preview.locator('h1')).to_have_text('实验记录')
        expect(preview.locator('td').last).to_have_text('2')
        expect(preview.locator('input[type=checkbox]')).to_be_checked()
        expect(preview.locator('.katex')).to_have_count(2)
        expect(preview.locator('pre code')).to_contain_text('$literal$ and **plain text**')
        expect(preview.locator('#raw-markup-marker, img[src]')).to_have_count(0)
        page.locator('#images').set_input_files({
            'name': 'pixel.png', 'mimeType': 'image/png', 'buffer': PIXEL,
        })
        image = preview.locator('img[alt="本地图像"]')
        expect(image).to_be_visible()
        page.wait_for_function("document.querySelector('#preview img[alt=\"本地图像\"]')?.naturalWidth === 1")
        edited = SOURCE.replace('实验记录', '修订稿')
        source.fill(edited)
        expect(preview.locator('h1')).to_have_text('修订稿')
        page.reload()
        expect(source).to_have_value(edited)
        expect(preview.locator('h1')).to_have_text('修订稿')
        with page.expect_download() as saving:
            page.locator('#download').click()
        download = saving.value
        assert download.suggested_filename == '笔记.md'
        assert Path(download.path()).read_bytes() == file_bytes(edited)

        page.set_viewport_size({'width': 320, 'height': 800})
        page.emulate_media(color_scheme='dark')
        page.locator('button[data-view="preview"]').click()
        expect(source).to_be_hidden()
        expect(preview).to_be_visible()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
        assert not unexpected and not errors, (unexpected, errors)
        browser.close()


def main():
    with ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=ROOT)) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            check_editor(f'http://127.0.0.1:{server.server_port}')
        finally:
            server.shutdown()
            thread.join()
    print('PASS Markdown import, live preview, local image, draft recovery and UTF-8 export')


if __name__ == '__main__':
    main()
