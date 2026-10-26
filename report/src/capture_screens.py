"""Capture the web-app screenshots used in the report (needs the server on port 8000).

Usage (from the project root, with the app running):
    python report/src/capture_screens.py

Uses Playwright with the installed Microsoft Edge. Writes JPEGs to report/figures/screens/.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "figures" / "screens"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://127.0.0.1:8000"


def shot(page, name: str, full: bool = False) -> None:
    png = OUT / f"{name}.png"
    page.screenshot(path=str(png), full_page=full)
    Image.open(png).convert("RGB").save(OUT / f"{name}.jpg", quality=86, optimize=True)
    png.unlink()
    print("saved", name)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge")
    page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)

    page.goto(f"{BASE}/#/studio")
    page.wait_for_selector("button.sample")
    page.wait_for_timeout(800)
    shot(page, "s01_upload")

    page.locator("button.sample", has_text="Handwritten note").click()
    page.wait_for_selector(".mem", timeout=60_000)
    page.wait_for_timeout(2500)
    shot(page, "s02_processing")
    page.wait_for_selector(".wgrid", timeout=240_000)
    page.wait_for_timeout(800)
    page.locator(".ln .dis").first.click()
    page.wait_for_timeout(300)
    shot(page, "s03_results_consensus")
    page.keyboard.press("Escape")
    page.locator(".ttab", has_text="Review").click()
    page.wait_for_timeout(300)
    shot(page, "s04_review")

    # Phone layout of the same result.
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_timeout(500)
    shot(page, "m1_document")
    page.locator(".mobile-pane-switch").get_by_role("radio", name="Text").click()
    page.locator(".ttab", has_text="Lines").click()
    page.wait_for_timeout(300)
    page.evaluate("window.scrollTo(0, 520)")
    page.wait_for_timeout(300)
    shot(page, "m2_lines")
    page.set_viewport_size({"width": 1440, "height": 900})

    page.get_by_role("button", name="New document").click()
    page.wait_for_selector("button.sample")
    page.locator("button.sample", has_text="Application form").click()
    page.wait_for_selector(".wgrid", timeout=240_000)
    page.wait_for_timeout(800)
    shot(page, "s05_form_layout")

    page.goto(f"{BASE}/#/compare")
    page.wait_for_selector(".cmp-sample")
    page.locator(".cmp-sample", has_text="Handwritten note").click()
    page.wait_for_function("document.querySelectorAll('.col').length > 0 && document.querySelectorAll('.col .col-loading').length === 0",
                           timeout=420_000)
    page.wait_for_timeout(800)
    shot(page, "s06_compare")

    page.goto(f"{BASE}/#/research")
    page.wait_for_selector(".kpis")
    page.wait_for_timeout(900)
    shot(page, "s07_research")
    page.locator("#voting").scroll_into_view_if_needed()
    page.evaluate("window.scrollBy(0, -70)")
    page.wait_for_timeout(500)
    shot(page, "s08_research_voting")

    page.goto(f"{BASE}/#/how-it-works")
    page.wait_for_selector(".timeline")
    page.wait_for_timeout(500)
    shot(page, "s09_how")
    browser.close()

# Put the two phone screenshots side by side.
left, right = Image.open(OUT / "m1_document.jpg"), Image.open(OUT / "m2_lines.jpg")
gap = 40
pair = Image.new("RGB", (left.width + right.width + gap, max(left.height, right.height)), "white")
pair.paste(left, (0, 0))
pair.paste(right, (left.width + gap, 0))
pair.save(OUT / "s10_mobile.jpg", quality=86, optimize=True)
print("saved s10_mobile")
