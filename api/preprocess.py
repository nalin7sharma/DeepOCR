"""Page-level preprocessing for the web pipeline (with visual stage previews).

Steps (any subset, applied in this order):
  * ``deskew``   — projection-profile skew estimation (±12°) and rotation.
  * ``denoise``  — Gaussian + median filtering (``ImageProcessor.remove_noise``).
  * ``contrast`` — CLAHE local contrast enhancement (``ImageProcessor.enhance_contrast``).
  * ``binarize`` — adaptive (local mean) thresholding for uneven illumination.

The OCR engines receive the processed page; region coordinates therefore refer
to the processed page, which is also what the UI shows as the page preview.
"""
from __future__ import annotations

import base64
import time

import cv2
import numpy as np

from preprocessing.image_processor import ImageProcessor

STEP_ORDER = ("deskew", "denoise", "contrast", "binarize")
STEP_LABELS = {
    "original": "Original",
    "deskew": "Deskewed",
    "denoise": "Denoised",
    "contrast": "Contrast (CLAHE)",
    "binarize": "Binarized",
}


def encode_preview(image: np.ndarray, max_side: int = 900, quality: int = 70) -> str:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(encoded.tobytes()).decode("ascii") if ok else ""


def estimate_skew(image: np.ndarray, max_angle: float = 12.0, step: float = 0.25) -> float:
    """Return the rotation (degrees) that makes text lines horizontal.

    Text rows produce sharp peaks in the horizontal projection profile when
    aligned; the angle maximising the profile's variance is selected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    scale = min(1.0, 900 / max(h, w))
    small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ink = _ink_mask(small)
    if np.count_nonzero(ink) < 50:
        return 0.0
    sh, sw = ink.shape
    center = (sw / 2, sh / 2)

    def score(angle: float) -> float:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(ink, matrix, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        profile = rotated.sum(axis=1, dtype=np.float64)
        return float(np.var(profile))

    coarse = np.arange(-max_angle, max_angle + 1e-6, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(max(-max_angle, best - 1.0), min(max_angle, best + 1.0) + 1e-6, step)
    best = max(fine, key=score)
    return float(round(best, 2))


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary text mask that is robust to shadows, uneven lighting and speckle noise.

    The paper is estimated with a grey-level dilation (a max filter wider than
    any stroke) and divided out (flat-field correction) before Otsu
    thresholding, so a dark shading gradient is not mistaken for ink.
    """
    if float(np.median(gray)) < 127:  # light text on a dark background
        gray = 255 - gray
    k = max(15, (min(gray.shape[:2]) // 25) | 1)
    paper = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    paper = cv2.blur(paper, (k, k))
    flat = cv2.divide(gray, np.maximum(paper, 1), scale=255)
    flat = cv2.medianBlur(flat, 3)
    ink = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def rotate_page(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate with an enlarged canvas, filling with the page background colour."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2 - w / 2
    matrix[1, 2] += new_h / 2 - h / 2
    border = tuple(int(v) for v in np.median(image.reshape(-1, image.shape[2] if image.ndim == 3 else 1), axis=0))
    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def binarize(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    block = max(15, (min(gray.shape[:2]) // 40) | 1)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 12)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def preprocess_page(image: np.ndarray, steps: list[str], with_previews: bool = True) -> tuple[np.ndarray, dict]:
    """Apply the requested steps; return the processed BGR page and a report."""
    requested = [step for step in STEP_ORDER if step in set(steps)]
    report: dict = {"applied": [], "deskew_angle": None, "stages": []}
    if with_previews and requested:
        report["stages"].append({"id": "original", "label": STEP_LABELS["original"], "ms": 0.0,
                                 "image": encode_preview(image)})
    current = image
    for step in requested:
        started = time.perf_counter()
        if step == "deskew":
            angle = estimate_skew(current)
            report["deskew_angle"] = angle
            if abs(angle) >= 0.3:
                current = rotate_page(current, angle)
        elif step == "denoise":
            current = ImageProcessor.remove_noise(current)
        elif step == "contrast":
            current = ImageProcessor.enhance_contrast(current)
        elif step == "binarize":
            current = binarize(current)
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        report["applied"].append(step)
        if with_previews:
            stage = {"id": step, "label": STEP_LABELS[step], "ms": elapsed, "image": encode_preview(current)}
            if step == "deskew":
                stage["label"] = f"Deskewed ({report['deskew_angle']:+.2f}°)"
            report["stages"].append(stage)
    return np.ascontiguousarray(current), report
