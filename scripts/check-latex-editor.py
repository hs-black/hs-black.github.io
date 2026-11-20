#!/usr/bin/env python3
"""Exercise the static LaTeX editor. Requires Playwright, Chromium, and pdftotext."""
from codecs import BOM_UTF8
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess
from threading import Thread
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
SOURCE = r"""\documentclass{article}
% 中文笔记
\begin{document}
\section{Energy}
Estimate $x^2$.
\input{chapter}
See Section~\ref{sec:mass}.
\end{document}
"""
CHAPTER = r"""\section{Inertial mass}\label{sec:mass}
Evidence from a local attachment.
"""


def source_bytes(text):
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

        def allow_static_resources(route):
            url = route.request.url
            if urlsplit(url).scheme in ('http', 'https') and (
                not url.startswith(origin + '/') or route.request.method not in ('GET', 'HEAD')
            ):
                unexpected.append((route.request.method, url))
                route.abort()
            else:
                route.continue_()

        page.context.route('**/*', allow_static_resources)
        page.goto(origin + '/tools/latex-editor/')
        page.locator('#file').set_input_files({
            'name': '实验.TeX', 'mimeType': 'text/plain', 'buffer': source_bytes(SOURCE),
        })
        editor = page.locator('#source')
        preview = page.locator('#preview')
        expect(editor).to_have_value(SOURCE)
        expect(preview.locator('.katex')).to_have_count(1)
        editor.fill(SOURCE.replace('$x^2$', r'$\frac{x^2$'))
        expect(preview.locator('.tex-math-error')).to_be_visible()
        edited = SOURCE.replace('Estimate', 'Calculate')
        editor.fill(edited)
        expect(preview.locator('.tex-math-error')).to_have_count(0)
        expect(preview).to_contain_text('Calculate')
        with page.expect_download() as saving:
            page.locator('#download').click()
        download = saving.value
        assert download.suggested_filename == '实验.TeX'
        assert Path(download.path()).read_bytes() == source_bytes(edited)

        page.locator('#attachments').set_input_files({
            'name': 'chapter.tex', 'mimeType': 'text/plain', 'buffer': CHAPTER.encode(),
        })
        page.locator('#compile').click()
        pdf = page.locator('#pdf-download')
        expect(pdf).to_be_visible(timeout=90000)
        with page.expect_download() as saving:
            pdf.click()
        result = saving.value
        assert result.suggested_filename == '实验.pdf'
        text = subprocess.check_output(['pdftotext', '-layout', str(result.path()), '-'], text=True)
        assert 'Calculate' in text and 'Evidence from a local attachment.' in text, text
        assert 'See Section 2.' in text and '??' not in text, text
        assert not errors and not unexpected, (errors, unexpected)
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
    print('PASS LaTeX import, live preview, UTF-8 source export, and local PDF with resolved references')


if __name__ == '__main__':
    main()
