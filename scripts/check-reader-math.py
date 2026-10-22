#!/usr/bin/env python3
"""Browser regression checks for ReaderMath; no external requests or artifacts.

Dependencies: Python 3.10+ and Playwright (python -m pip install playwright).
Install a browser with `python -m playwright install chromium`, use an installed
Chromium/Chrome on PATH, or set CHROMIUM_PATH to its executable.

Run from any directory: python /path/to/repo/scripts/check-reader-math.py
The script serves this checkout on an ephemeral localhost port and shuts down
the server and browser on completion. It does not modify the checkout.
"""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
from threading import Thread

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "/scripts/fixtures/reader-math.html"
KATEX = ROOT / "assets/vendor/katex/0.16.11"


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def chromium_path(playwright):
    override = os.environ.get("CHROMIUM_PATH")
    if override:
        candidate = shutil.which(override)
        if candidate:
            return candidate
        raise SystemExit("CHROMIUM_PATH must name an executable Chromium browser.")
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        candidate = shutil.which(name)
        if candidate:
            return candidate
    bundled = Path(playwright.chromium.executable_path)
    if bundled.is_file():
        return str(bundled)
    raise SystemExit("Chromium not found. Run: python -m playwright install chromium")


def within_viewport(page, width, height):
    rect = page.locator("#reader-symbol-tip").bounding_box()
    assert rect, "The symbol explanation must be visible."
    assert 0 <= rect["x"] <= rect["x"] + rect["width"] <= width + 0.5, rect
    assert 0 <= rect["y"] <= rect["y"] + rect["height"] <= height + 0.5, rect


def run_checks(browser, url):
    errors = []
    context = browser.new_context(viewport={"width": 1100, "height": 850}, reduced_motion="reduce")

    def open_page(ctx):
        page = ctx.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url, wait_until="load")
        page.wait_for_function("typeof ReaderMath === 'object'")
        return page

    page = open_page(context)
    expect(page.locator(".katex-error")).to_have_count(0)
    expect(page.locator("#global .reader-symbol")).to_have_count(1)
    expect(page.locator("#plain")).to_have_attribute("role", "button")
    expect(page.locator("#unknown .katex")).to_have_count(1)
    expect(page.locator("#unknown [role=button], #unknown [tabindex]")).to_have_count(0)
    # Trust is limited to one valid data-symbol key, never arbitrary HTML hooks.
    expect(page.locator("#unsafe a, #unsafe [data-symbol], #unsafe [data-extra]")).to_have_count(0)
    expect(page.locator('#unsafe [style*="position:fixed"]')).to_have_count(0)
    # Unknown commands can appear red without a .katex-error in graceful mode.
    parse_errors = page.locator('annotation[encoding="application/x-tex"]').evaluate_all("""
        annotations => annotations.flatMap(node => {
            try {
                katex.renderToString(node.textContent, {throwOnError: true, strict: 'ignore'});
                return [];
            } catch (error) {
                return [{tex: node.textContent, error: error.message}];
            }
        })
    """)
    assert not parse_errors, f"Invalid formula sources: {parse_errors}"
    print("PASS rendering, unknown symbols, and the KaTeX trust allowlist")

    tip = page.locator("#reader-symbol-tip")
    for selector, expected in (
        ("#global .reader-symbol", "条请求"),
        ("#scoped .reader-symbol", "token 总数"),
        ("#inherited [data-symbol=b]", "token 总数"),
        ("#inherited [data-symbol=bw]", "HBM 带宽"),
    ):
        page.locator(selector).hover()
        expect(tip).to_be_visible()
        expect(tip).to_contain_text(expected)
    print("PASS global, local, and inherited scope definitions")

    symbol = page.locator("#global .reader-symbol")
    symbol.focus()
    expect(tip).to_be_visible()
    expect(symbol).to_have_attribute("aria-describedby", "reader-symbol-tip")
    assert page.locator(".reader-symbol").evaluate_all("""elements => elements.every(el => {
        for (let ancestor = el; ancestor; ancestor = ancestor.parentElement) {
            if (ancestor.getAttribute('aria-hidden') === 'true') return false;
        }
        return true;
    })"""), "No interactive symbol may have an aria-hidden ancestor."
    expect(page.locator("#formula .katex-mathml math")).to_have_count(1)
    expect(page.locator("#formula .katex-mathml annotation")).to_have_text(
        r"T=\frac{2\htmlData{symbol=b}{B}DF}{C}"
    )
    symbol.press("Enter")
    expect(symbol).to_have_attribute("aria-expanded", "true")
    symbol.press(" ")
    expect(tip).to_be_hidden()
    symbol.press("Enter")
    expect(tip).to_be_visible()
    symbol.press("Escape")
    expect(tip).to_be_hidden()
    expect(symbol).to_have_attribute("aria-expanded", "false")
    assert symbol.get_attribute("aria-describedby") is None
    print("PASS preserved MathML, focus accessibility, keyboard toggle, and Escape")

    # Explicit HTML can render inside SVG; native SVG text must remain untouched,
    # even when callers pass the SVG itself as the render root.
    expect(page.locator("#svgmath .katex")).to_have_count(1)
    page.evaluate("ReaderMath.render(document.querySelector('svg'))")
    expect(page.locator("#rawsvg")).to_have_text(r"\(raw\)")
    expect(page.locator("#rawsvg .katex")).to_have_count(0)
    count = page.locator(".katex").count()
    page.evaluate("ReaderMath.render(document); ReaderMath.render(document.body)")
    expect(page.locator(".katex")).to_have_count(count)
    for target in ("dynamic", "svgmath"):
        page.evaluate("""id => {
            const el = document.getElementById(id);
            el.dataset.tex = '\\\\frac{3}{4}';
            ReaderMath.render(el);
            ReaderMath.render(el);
        }""", target)
        expect(page.locator(f"#{target} .katex")).to_have_count(1)
        expect(page.locator(f"#{target} annotation")).to_have_text(r"\frac{3}{4}")
    print("PASS dynamic HTML/foreignObject math, native SVG exclusion, and idempotence")

    for width in (320, 390):
        page.set_viewport_size({"width": width, "height": 720})
        page.emulate_media(color_scheme="dark", reduced_motion="reduce")
        page.evaluate("ReaderMath.hide(); document.activeElement?.blur()")
        symbol.focus()
        expect(tip).to_be_visible()
        within_viewport(page, width, 720)
    print("PASS explanation bounds at 320/390px in dark mode")

    touch = browser.new_context(viewport={"width": 390, "height": 760}, is_mobile=True, has_touch=True)
    phone = open_page(touch)
    phone_symbol = phone.locator("#global .reader-symbol")
    phone_tip = phone.locator("#reader-symbol-tip")
    phone_symbol.tap()
    expect(phone_tip).to_be_visible()
    phone_symbol.tap()
    expect(phone_tip).to_be_hidden()
    phone.locator("#scoped .reader-symbol").tap()
    expect(phone_tip).to_be_visible()
    phone.touchscreen.tap(380, 740)
    expect(phone_tip).to_be_hidden()
    print("PASS touch toggle and outside dismissal")

    missing = browser.new_context()
    missing.route("**/katex.min.js", lambda route: route.abort())
    fallback = open_page(missing)
    expect(fallback.locator("#global")).to_contain_text(r"\htmlData")
    expect(fallback.locator("#dynamic")).to_have_text(r"\frac{1}{2}")
    expect(fallback.locator(".katex")).to_have_count(0)
    # Auto-render captures KaTeX when loaded, so retry both dependencies in order.
    fallback.add_script_tag(path=str(KATEX / "katex.min.js"))
    fallback.add_script_tag(path=str(KATEX / "contrib/auto-render.min.js"))
    fallback.evaluate("ReaderMath.render(document.body)")
    expect(fallback.locator("#global .reader-symbol")).to_have_count(1)
    expect(fallback.locator("#dynamic .katex")).to_have_count(1)
    expect(fallback.locator(".katex-error")).to_have_count(0)
    assert not errors, f"Uncaught browser errors: {errors}"
    print("PASS dependency failure preserves source, and retry restores rendering")
    for ctx in (context, touch, missing):
        ctx.close()


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=chromium_path(playwright))
            try:
                run_checks(browser, f"http://127.0.0.1:{server.server_port}{FIXTURE}")
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
