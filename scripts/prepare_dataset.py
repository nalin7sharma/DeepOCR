#!/usr/bin/env python
"""
Dataset Preparation Script for DeepOCR.

IAM (real handwriting):
  python scripts/download_iam_hf.py          # recommended: official splits from the HF Hub
  python scripts/prepare_dataset.py --dataset iam   # validate a manual IAM download

Synthetic (printed + handwriting-style fonts with realistic degradations):
  python scripts/prepare_dataset.py --dataset synthetic --n_samples 6500

The synthetic generator renders text lines with many system fonts (printed
serif/sans/monospace and script/handwriting-style faces) and applies document
degradations — paper tone, uneven illumination, Gaussian noise, blur, JPEG
artefacts, slight rotation, ink thickness changes, low contrast and inverted
(dark-mode) rendering.  Text comes from IAM *training* transcriptions when
available (never validation/test) plus document-style templates such as
dates, amounts, phone numbers, e-mails and form labels.

Output layout (explicit splits, read by scripts/train.py)::

    data/processed/synthetic/
        images/{train,val,test}/000001.png
        train.txt / val.txt / test.txt   (relative_path<TAB>text)
        meta.tsv                          (path, split, font, style)
"""
from __future__ import annotations

import argparse
import random
import string
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.line_image import normalize_transcription
from utils.config_loader import load_all_configs, get_project_root
from utils.logger import get_logger

logger = get_logger(__name__)

PRINTED_FONTS = [
    "arial.ttf", "arialbd.ttf", "ariali.ttf", "ARIALN.TTF", "times.ttf", "timesbd.ttf", "timesi.ttf",
    "calibri.ttf", "calibrib.ttf", "cambria.ttc", "georgia.ttf", "georgiai.ttf", "verdana.ttf",
    "tahoma.ttf", "trebuc.ttf", "consola.ttf", "cour.ttf", "BOOKOS.TTF", "segoeui.ttf",
    "DejaVuSans.ttf", "DejaVuSerif.ttf", "DejaVuSansMono.ttf", "LiberationSans-Regular.ttf",
    "LiberationSerif-Regular.ttf",
]
SCRIPT_FONTS = [
    "Inkfree.ttf", "segoesc.ttf", "segoepr.ttf", "comic.ttf", "comici.ttf", "LHANDW.TTF",
    "BRADHITC.TTF", "FREESCPT.TTF", "MISTRAL.TTF", "ITCKRIST.TTF", "VINERITC.TTF",
    "JUICE___.TTF", "PRISTINA.TTF",
]
FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"),
             Path("/usr/share/fonts/truetype/liberation"), Path("/Library/Fonts")]

BUILTIN_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "Deep learning has improved text recognition significantly.",
    "Handwritten text is more challenging than printed text.",
    "Image preprocessing improves recognition accuracy.",
    "Please sign and return the attached form by Friday.",
    "Meeting is scheduled at 10 AM in the main conference room.",
    "Thank you for your prompt reply to our enquiry.",
    "Total amount due must be paid within thirty days.",
]
FIRST_NAMES = ["Aarav", "Priya", "John", "Maria", "Wei", "Fatima", "Rohan", "Emma", "Liam", "Sara", "Nalin", "Ananya"]
LAST_NAMES = ["Sharma", "Patel", "Smith", "Garcia", "Chen", "Khan", "Gupta", "Brown", "Wilson", "Iyer"]
LABELS = ["Name:", "Date:", "Address:", "Phone:", "Email:", "Amount:", "Invoice No:", "Signature:",
          "Date of Birth:", "Roll No:", "Department:", "Total:", "Account:", "Ref:", "Subject:"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def _resolve_fonts(names: list[str]) -> list[Path]:
    found = []
    for name in names:
        for directory in FONT_DIRS:
            candidate = directory / name
            if candidate.exists():
                found.append(candidate)
                break
    return found


def _template_text(rng: random.Random, words: list[str]) -> str:
    kind = rng.randrange(9)
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    if kind == 0:
        return f"{rng.choice(LABELS)} {name}"
    if kind == 1:
        return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1990, 2026)}"
    if kind == 2:
        return f"{rng.choice(MONTHS)} {rng.randint(1, 28)}, {rng.randint(1990, 2026)}"
    if kind == 3:
        return f"Total: ${rng.randint(1, 9999):,}.{rng.randint(0, 99):02d}"
    if kind == 4:
        return f"+91 {rng.randint(70000, 99999)} {rng.randint(10000, 99999)}"
    if kind == 5:
        user = name.lower().replace(" ", ".")
        return f"{user}@{rng.choice(['gmail.com', 'example.org', 'college.edu'])}"
    if kind == 6:
        return f"INV-{rng.randint(1000, 99999)} ({rng.randint(1, 99)}% paid)"
    if kind == 7:
        return f"{rng.randint(1, 999)} {rng.choice(words).title()} Street, {rng.choice(words).title()}"
    return f"{rng.choice(LABELS)} {' '.join(rng.choice(words) for _ in range(rng.randint(1, 4)))}"


def _text_sources(root: Path) -> tuple[list[str], list[str]]:
    iam_train = root / "data" / "processed" / "iam" / "train.txt"
    sentences: list[str] = []
    if iam_train.exists():
        for line in iam_train.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[1].strip():
                sentences.append(parts[1].strip())
    sentences.extend(BUILTIN_SENTENCES)
    words = [w.strip(string.punctuation) for s in sentences for w in s.split()]
    words = [w for w in words if len(w) > 2 and w.isalpha()] or ["text"]
    return sentences, words


def _sample_text(rng: random.Random, sentences: list[str], words: list[str], charset: set[str]) -> str:
    if rng.random() < 0.35:
        text = _template_text(rng, words)
    else:
        tokens = rng.choice(sentences).split()
        n = rng.randint(2, min(10, max(2, len(tokens))))
        start = rng.randint(0, max(0, len(tokens) - n))
        text = " ".join(tokens[start:start + n])
        if rng.random() < 0.15:
            text = text.upper()
    text = "".join(c for c in normalize_transcription(text) if c in charset)
    return " ".join(text.split())[:80]


def _render(text: str, font_path: Path, rng: random.Random) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    size = rng.randint(30, 52)
    font = ImageFont.truetype(str(font_path), size)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    pad_x, pad_y = rng.randint(4, 30), rng.randint(4, 16)
    width, height = right - left + 2 * pad_x, bottom - top + 2 * pad_y
    paper = rng.randint(185, 255)
    ink = rng.randint(0, 90)
    image = Image.new("L", (max(8, width), max(8, height)), color=paper)
    ImageDraw.Draw(image).text((pad_x - left, pad_y - top), text, fill=ink, font=font)
    return np.array(image)


def _degrade(image: np.ndarray, rng: random.Random) -> np.ndarray:
    img = image.astype(np.float32)
    h, w = img.shape
    if rng.random() < 0.45:  # uneven illumination / shadow gradient
        direction = rng.choice([np.linspace(0, 1, w)[None, :], np.linspace(0, 1, h)[:, None]])
        img -= direction * rng.uniform(10, 70)
    if rng.random() < 0.2:  # ink thickness (bleed / faded strokes)
        kernel = np.ones((2, 2), np.uint8)
        op = cv2.erode if rng.random() < 0.5 else cv2.dilate
        img = op(img, kernel, iterations=1)
    if rng.random() < 0.3:  # small skew
        angle = rng.uniform(-2.0, 2.0)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.35:  # defocus blur
        sigma = rng.uniform(0.5, 1.6)
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    if rng.random() < 0.15:  # low contrast scan
        mean = img.mean()
        img = mean + (img - mean) * rng.uniform(0.35, 0.6)
    noise_sigma = rng.choice([0, 0, 3, 6, 10, 16])
    if noise_sigma:
        img += np.random.normal(0, noise_sigma, img.shape)
    if rng.random() < 0.08:  # salt & pepper specks
        mask = np.random.random(img.shape)
        img[mask < 0.004] = 0
        img[mask > 0.996] = 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:  # JPEG compression artefacts
        quality = rng.randint(20, 80)
        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            img = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if rng.random() < 0.1:  # dark-mode screenshot
        img = 255 - img
    return img


def prepare_synthetic(root: Path, cfg: dict, n_samples: int, seed: int = 42) -> None:
    """Generate the synthetic printed/handwriting-font dataset with explicit splits."""
    rng = random.Random(seed)
    np.random.seed(seed)
    charset = set(cfg.get("dataset", {}).get("charset", string.printable))
    printed, script = _resolve_fonts(PRINTED_FONTS), _resolve_fonts(SCRIPT_FONTS)
    if not printed and not script:
        logger.error("No TrueType fonts found. Install DejaVu/Liberation fonts or run on Windows.")
        sys.exit(1)
    logger.info(f"Fonts: {len(printed)} printed, {len(script)} handwriting-style")
    sentences, words = _text_sources(root)

    out_dir = root / "data" / "processed" / "synthetic"
    n_val = max(1, int(n_samples * 0.08))
    n_test = max(1, int(n_samples * 0.15))
    split_sizes = {"train": n_samples - n_val - n_test, "val": n_val, "test": n_test}
    meta_rows = ["path\tsplit\tfont\tstyle"]

    for split, count in split_sizes.items():
        image_dir = out_dir / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for index in range(1, count + 1):
            text = _sample_text(rng, sentences, words, charset)
            if not text:
                continue
            use_script = script and (not printed or rng.random() < 0.35)
            font_path = rng.choice(script if use_script else printed)
            try:
                image = _degrade(_render(text, font_path, rng), rng)
            except OSError:
                continue
            relative = f"images/{split}/{index:06d}.png"
            cv2.imwrite(str(out_dir / relative), image)
            rows.append(f"{relative}\t{text}")
            meta_rows.append(f"{relative}\t{split}\t{font_path.name}\t{'script' if use_script else 'printed'}")
        (out_dir / f"{split}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        logger.info(f"Synthetic {split}: {len(rows)} lines")
    (out_dir / "meta.tsv").write_text("\n".join(meta_rows) + "\n", encoding="utf-8")
    logger.info(f"Synthetic dataset written to {out_dir}")


def prepare_iam(root: Path, cfg: dict) -> None:
    """Validate the IAM export (HF download) or a manual IAM lines.txt download."""
    processed = root / "data" / "processed" / "iam"
    if (processed / "train.txt").exists():
        for split in ("train", "val", "test"):
            path = processed / f"{split}.txt"
            count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if path.exists() else 0
            logger.info(f"IAM {split}: {count} lines ({path})")
        return

    iam_cfg = cfg.get("dataset", {}).get("iam", {})
    gt_file = root / iam_cfg.get("gt_file", "data/raw/IAM/ascii/lines.txt")
    if not gt_file.exists():
        logger.error(
            "IAM data not found. Run: python scripts/download_iam_hf.py\n"
            "(or place the official lines.tgz/ascii.tgz extraction in data/raw/IAM/)."
        )
        return
    count = sum(1 for line in gt_file.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#") and " ok " in line)
    logger.info(f"IAM (manual download): {count} lines marked ok in {gt_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OCR dataset")
    parser.add_argument("--dataset", default="synthetic", choices=["iam", "synthetic"])
    parser.add_argument("--n_samples", type=int, default=6500,
                        help="Number of synthetic samples to generate (all splits)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_all_configs()
    root = get_project_root()

    if args.dataset == "iam":
        prepare_iam(root, cfg)
    else:
        prepare_synthetic(root, cfg, args.n_samples, args.seed)


if __name__ == "__main__":
    main()
