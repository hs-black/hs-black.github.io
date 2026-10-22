#!/usr/bin/env python3
"""Test article figure controls. Requires Python Playwright and Chromium.

Run: python3 scripts/check-article-steps.py
Uses a temporary localhost server and writes no files. CHROMIUM_PATH can select
the browser; otherwise use Chromium on PATH or Playwright's installed browser.
"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
from threading import Thread

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def main():
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(ROOT)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            executable = (shutil.which(os.environ.get('CHROMIUM_PATH', 'chromium'))
                          or shutil.which('chromium-browser') or p.chromium.executable_path)
            browser = p.chromium.launch(executable_path=executable)
            try:
                page = browser.new_page(viewport={'width': 800, 'height': 700})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}/scripts/fixtures/article-steps.html')
                page.wait_for_timeout(250)
                assert page.evaluate('figure.position === 0 && !figure.playing && figure === sameFigure')
                expect(page.locator('.article-step-controls')).to_have_count(1)
                expect(page.locator('[data-action=prev]')).to_be_disabled()
                page.locator('[data-action=next]').click()
                expect(page.locator('#used')).to_have_attribute('width', '100')
                page.locator('input[type=range]').fill('1.5')
                expect(page.locator('#used')).to_have_attribute('width', '200')
                page.locator('[data-action=prev]').click()
                expect(page.locator('#used')).to_have_attribute('width', '100')
                page.locator('[data-action=next]').click()
                expect(page.locator('[data-action=next]')).to_be_disabled()
                page.locator('[data-action=reset]').click()
                expect(page.locator('#used')).to_have_attribute('width', '0')
                page.locator('[data-action=play]').click()
                page.wait_for_function('figure.position === 2 && !figure.playing')
                expect(page.locator('#value')).to_have_text('已用 3 块')
                page.locator('[data-action=reset]').click()
                page.emulate_media(reduced_motion='reduce', color_scheme='dark')
                page.set_viewport_size({'width': 320, 'height': 700})
                page.locator('[data-action=play]').click()
                assert page.evaluate('figure.position === 1 && !figure.playing')
                assert page.evaluate('document.body.scrollWidth <= innerWidth')
                page.locator('input[type=range]').focus()
                page.keyboard.press('End')
                assert page.evaluate('figure.position === 2')
                page.keyboard.press('Home')
                assert page.evaluate('figure.position === 0')
                assert not errors, errors
                print('PASS paused start, independent state, scrub/rewind/reset, playback, reduced motion, keyboard and mobile layout')
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == '__main__':
    main()
