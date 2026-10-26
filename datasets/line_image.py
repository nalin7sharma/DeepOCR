"""
Shared text-line image normalisation for training, evaluation and inference.

Every recogniser in DeepOCR sees line images prepared by exactly the same
function, so a model trained on IAM / synthetic lines receives the same kind
of input when it is used on a crop detected inside a real document.

Steps:
  1. Grayscale conversion.
  2. Polarity normalisation — ink becomes bright on a dark background
     (handles both dark-on-light scans and light-on-dark screenshots).
  3. Brightness / contrast normalisation — background is subtracted and the
     ink intensity is stretched to the full 0-255 range.
  4. Aspect-preserving resize to the model's input height.
  5. Width fitting — right-pad with background, or squeeze very long lines.
"""
from __future__ import annotations

import re

import cv2
import numpy as np
import torch

_SPACE_BEFORE = re.compile(r"\s+([,.;:?!)\]])")
_SPACE_AFTER = re.compile(r"([(\[])\s+")
_CONTRACTION = re.compile(r"\s+'(s|t|re|ve|ll|d|m)\b")


def normalize_transcription(text: str) -> str:
    """Canonical transcription form used for training targets and scoring.

    IAM ground truth is tokenised ("Exchange ." / "we 've"); printed text is
    not.  Both are mapped to natural spacing so models and metrics agree.
    """
    text = " ".join(str(text).split())
    text = _SPACE_BEFORE.sub(r"\1", text)
    text = _SPACE_AFTER.sub(r"\1", text)
    text = _CONTRACTION.sub(r"'\1", text)
    return text.strip()


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a uint8 single-channel image."""
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def normalize_line(image: np.ndarray) -> np.ndarray:
    """Convert a line image to bright ink on a black background (uint8)."""
    gray = to_gray(image)
    if gray.size == 0:
        return gray
    # The background dominates a text line, so the median estimates it.
    if float(np.median(gray)) > 127:
        gray = 255 - gray
    work = gray.astype(np.float32)
    background = float(np.percentile(work, 50))
    work = np.clip(work - background, 0, None)
    peak = float(np.percentile(work, 99.5))
    if peak < 8:  # empty / blank crop: keep as is
        return np.zeros_like(gray)
    work = np.clip(work * (255.0 / peak), 0, 255)
    return work.astype(np.uint8)


def resize_to_height(image: np.ndarray, height: int, max_width: int | None = None) -> np.ndarray:
    """Resize preserving aspect ratio so the image is ``height`` pixels tall."""
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((height, max(1, height)), dtype=np.uint8)
    new_w = max(1, int(round(w * height / h)))
    if max_width is not None:
        new_w = min(new_w, max_width)
    interpolation = cv2.INTER_AREA if h > height else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, height), interpolation=interpolation)


def fit_width(image: np.ndarray, width: int) -> np.ndarray:
    """Right-pad with black (background) or squeeze to exactly ``width``."""
    h, w = image.shape[:2]
    if w > width:
        return cv2.resize(image, (width, h), interpolation=cv2.INTER_AREA)
    if w == width:
        return image
    canvas = np.zeros((h, width), dtype=image.dtype)
    canvas[:, :w] = image
    return canvas


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """uint8 [H, W] -> float tensor [1, H, W] in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(image)).float().div_(255.0).unsqueeze(0)


def prepare_line(image: np.ndarray, height: int, width: int) -> torch.Tensor:
    """Full inference-time preparation of one line crop -> [1, H, W] tensor."""
    normalized = normalize_line(image)
    resized = resize_to_height(normalized, height)
    return to_tensor(fit_width(resized, width))
