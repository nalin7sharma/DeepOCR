#!/usr/bin/env python
"""
Generate the built-in demo documents shown in the web app's sample gallery.

  samples/handwritten_note.png   IAM *test* lines composed into a page (real handwriting)
  samples/application_form.png   printed form: fields, checkboxes, ruled table, signature
  samples/noisy_scan.jpg         skewed, noisy, unevenly lit photocopy of a printed letter
  samples/report.pdf             two-page PDF: printed report + handwritten page
  samples/manifest.json          gallery metadata (+ thumbnails)

Usage:
    python scripts/make_samples.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from api.preprocess import rotate_page  # noqa: E402

OUT = ROOT / "samples"
FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/truetype/liberation")]


def font(names: list[str], size: int) -> ImageFont.FreeTypeFont:
    for name in names:
        for directory in FONT_DIRS:
            path = directory / name
            if path.exists():
                return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


SANS = ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]
SANS_BOLD = ["arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"]
SERIF = ["georgia.ttf", "times.ttf", "DejaVuSerif.ttf"]
SCRIPT = ["Inkfree.ttf", "segoepr.ttf", "comic.ttf", "DejaVuSans.ttf"]


def thumb(source: Path, target: Path, width: int = 300) -> None:
    image = Image.open(source).convert("RGB")
    image.thumbnail((width, int(width * 1.5)))
    image.save(target, quality=82)


def handwritten_page(lines: int = 8, start: int = 1) -> Image.Image:
    test_dir = ROOT / "data" / "processed" / "iam"
    manifest = test_dir / "test.txt"
    if not manifest.exists():
        raise SystemExit("IAM test data missing: run python scripts/download_iam_hf.py first.")
    rows = [line.split("\t", 1) for line in manifest.read_text(encoding="utf-8").splitlines() if "\t" in line]
    crops = []
    for rel, _ in rows[start - 1:start - 1 + lines]:
        img = Image.open(test_dir / rel).convert("L")
        crops.append(img.resize((int(img.width * 0.62), int(img.height * 0.62))))
    width = max(c.width for c in crops) + 160
    height = sum(c.height for c in crops) + 40 * len(crops) + 160
    page = Image.new("L", (width, height), 250)
    y = 90
    rng = random.Random(3)
    for crop in crops:
        page.paste(crop, (80 + rng.randint(-10, 10), y))
        y += crop.height + 40
    paper = np.array(page, dtype=np.float32)
    paper -= np.linspace(0, 12, paper.shape[1])[None, :]
    return Image.fromarray(np.clip(paper, 0, 255).astype(np.uint8)).convert("RGB")


def application_form() -> Image.Image:
    W, H = 1240, 1600
    page = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(page)
    d.text((80, 70), "College Scholarship Application", font=font(SANS_BOLD, 46), fill="#111")
    d.text((80, 135), "Office of Student Affairs  |  Form SA-2026", font=font(SANS, 24), fill="#444")
    d.line((80, 180, W - 80, 180), fill="#111", width=3)
    body, hand = font(SANS, 28), font(SCRIPT, 34)
    fields = [("Name:", "Ananya Iyer"), ("Roll No:", "21CS1047"), ("Department:", "Computer Science"),
              ("Email:", "ananya.iyer@college.edu"), ("Date:", "26/10/2026")]
    y = 220
    for label, value in fields:
        d.text((80, y), label, font=body, fill="#111")
        d.line((300, y + 36, 900, y + 36), fill="#999", width=2)
        d.text((310, y - 6), value, font=hand, fill="#1d3a8a")
        y += 70
    y += 20
    d.text((80, y), "Scholarship type:", font=font(SANS_BOLD, 28), fill="#111")
    y += 55
    for checked, option in [(True, "Merit based"), (False, "Need based"), (True, "Sports quota")]:
        d.rectangle((90, y, 120, y + 30), outline="#111", width=3)
        if checked:
            d.line((95, y + 15, 104, y + 25), fill="#111", width=4)
            d.line((104, y + 25, 117, y + 4), fill="#111", width=4)
        d.text((140, y - 2), option, font=body, fill="#111")
        y += 52
    y += 30
    d.text((80, y), "Fee details", font=font(SANS_BOLD, 28), fill="#111")
    y += 50
    cols, rows = [80, 520, 800, 1160], ["Item|Semester|Amount", "Tuition|Odd 2026|$1,250.00", "Hostel|Odd 2026|$480.00", "Library|Annual|$35.00", "Total||$1,765.00"]
    row_h = 58
    for r, row in enumerate(rows):
        top = y + r * row_h
        for c, cell in enumerate(row.split("|")):
            d.text((cols[c] + 14, top + 13), cell, font=font(SANS_BOLD if r == 0 else SANS, 26), fill="#111")
    for r in range(len(rows) + 1):
        d.line((cols[0], y + r * row_h, cols[-1], y + r * row_h), fill="#111", width=2)
    for x in cols:
        d.line((x, y, x, y + len(rows) * row_h), fill="#111", width=2)
    y += len(rows) * row_h + 70
    d.text((80, y), "I confirm the information above is correct.", font=body, fill="#111")
    y += 80
    d.text((80, y), "Signature:", font=body, fill="#111")
    d.line((240, y + 36, 640, y + 36), fill="#999", width=2)
    d.text((260, y - 14), "Ananya", font=font(["FREESCPT.TTF", "segoesc.ttf"] + SCRIPT, 54), fill="#1d3a8a")
    return page


def noisy_scan() -> np.ndarray:
    W, H = 1240, 1100
    page = Image.new("RGB", (W, H), (246, 244, 236))
    d = ImageDraw.Draw(page)
    d.text((90, 80), "Notice of Annual Meeting", font=font(SERIF, 44), fill="#1a1a1a")
    paragraph = [
        "Dear Members,",
        "The annual general meeting of the Computer Society will be held",
        "on Friday, 14 November 2026 at 10:30 AM in the Main Auditorium.",
        "All members are requested to attend. The agenda includes the",
        "treasurer's report, election of new office bearers and plans for",
        "the upcoming technical festival. Light refreshments will be served.",
        "Please confirm your attendance by replying to this notice.",
        "Sincerely,",
        "The Secretary",
    ]
    y = 170
    for line in paragraph:
        d.text((90, y), line, font=font(SERIF, 30), fill="#202020")
        y += 66 if line.endswith(",") else 54
    img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    img = rotate_page(img, 4.2)
    h, w = img.shape[:2]
    shade = np.linspace(1.0, 0.72, w)[None, :, None] * np.linspace(1.0, 0.9, h)[:, None, None]
    img = img.astype(np.float32) * shade
    img += np.random.default_rng(7).normal(0, 13, img.shape)
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (0, 0), 0.9)
    return img


def report_pdf(path: Path, handwritten: Image.Image) -> None:
    import fitz

    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((60, 80), "Project Status Report", fontsize=24, fontname="helv")
    page.insert_text((60, 108), "Deep Learning OCR System - Week 12", fontsize=12, fontname="helv")
    text = (
        "Summary: the recognition models were trained on IAM handwriting and synthetic\n"
        "printed text. The CRNN and CNN-BiLSTM baselines are compared against a pretrained\n"
        "Transformer (TrOCR) using character and word error rates on held-out test sets.\n\n"
        "Next steps: improve layout analysis for multi-column documents and add Hindi\n"
        "handwriting support."
    )
    page.insert_textbox(fitz.Rect(60, 135, 540, 260), text, fontsize=11, fontname="helv")
    table_top, rows = 290, [("Model", "CER", "WER"), ("CNN-BiLSTM", "baseline", "baseline"), ("CRNN", "model 2", "model 2"), ("TrOCR", "advanced", "advanced")]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            page.insert_text((70 + c * 150, table_top + 22 + r * 28), cell, fontsize=11, fontname="helv")
    for r in range(len(rows) + 1):
        page.draw_line((60, table_top + r * 28), (510, table_top + r * 28))
    for c in range(4):
        page.draw_line((60 + c * 150, table_top), (60 + c * 150, table_top + len(rows) * 28))
    hand_path = OUT / "_page2.png"
    handwritten.save(hand_path)
    page2 = document.new_page(width=595, height=842)
    page2.insert_image(fitz.Rect(30, 40, 565, 800), filename=str(hand_path))
    document.save(path)
    document.close()
    hand_path.unlink()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    np.random.seed(0)
    manifest = []

    note = handwritten_page(8, start=1)
    note.save(OUT / "handwritten_note.png")
    thumb(OUT / "handwritten_note.png", OUT / "handwritten_note_thumb.jpg")
    manifest.append({"id": "handwritten", "file": "handwritten_note.png", "thumb": "handwritten_note_thumb.jpg",
                     "title": "Handwritten note", "subtitle": "Real IAM handwriting (unseen test writers)",
                     "kind": "handwriting", "suggested_engine": "crnn", "preprocess": []})

    form = application_form()
    form.save(OUT / "application_form.png")
    thumb(OUT / "application_form.png", OUT / "application_form_thumb.jpg")
    manifest.append({"id": "form", "file": "application_form.png", "thumb": "application_form_thumb.jpg",
                     "title": "Application form", "subtitle": "Fields, checkboxes, ruled table and signature",
                     "kind": "form", "suggested_engine": "paddle", "preprocess": []})

    cv2.imwrite(str(OUT / "noisy_scan.jpg"), noisy_scan(), [cv2.IMWRITE_JPEG_QUALITY, 55])
    thumb(OUT / "noisy_scan.jpg", OUT / "noisy_scan_thumb.jpg")
    manifest.append({"id": "noisy", "file": "noisy_scan.jpg", "thumb": "noisy_scan_thumb.jpg",
                     "title": "Noisy skewed scan", "subtitle": "4° skew, noise, blur and shadows",
                     "kind": "degraded", "suggested_engine": "paddle", "preprocess": ["deskew", "denoise", "contrast"]})

    report_pdf(OUT / "report.pdf", handwritten_page(6, start=40))
    import fitz
    with fitz.open(OUT / "report.pdf") as document:
        document[0].get_pixmap(matrix=fitz.Matrix(0.6, 0.6)).save(OUT / "report_thumb.png")
    manifest.append({"id": "report", "file": "report.pdf", "thumb": "report_thumb.png",
                     "title": "Two-page PDF report", "subtitle": "Printed page with a table, plus a handwritten page",
                     "kind": "pdf", "suggested_engine": "paddle", "preprocess": ["deskew"]})

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} samples to {OUT}")


if __name__ == "__main__":
    main()
