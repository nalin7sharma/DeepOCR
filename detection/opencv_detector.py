"""
OpenCV-based text region detector for DeepOCR.

Uses MSER (Maximally Stable Extremal Regions) + morphological post-processing
to find text bounding boxes.  This is a practical, lightweight baseline
that works without a trained DL detector.

The detector interface is designed so a DL-based detector (e.g., CRAFT, DBNet)
can drop in as a replacement by implementing BaseTextDetector.

Algorithm:
  1. Convert to grayscale, denoise, threshold.
  2. Run MSER to find character-level regions.
  3. Filter regions by aspect ratio and area.
  4. Dilate+merge nearby regions into line bounding boxes.
  5. Sort by reading order (top-to-bottom, left-to-right).
"""
from __future__ import annotations

import cv2
import numpy as np

from .base_detector import BaseTextDetector


class OpenCVTextDetector(BaseTextDetector):
    """Practical text detector using MSER + morphological grouping.

    Args:
        config: Optional dict with detection parameters.

    Config keys (defaults):
        min_area         (50)    Minimum MSER region area (px^2)
        max_area         (10000) Maximum MSER region area
        aspect_ratio_min (0.1)   Min bbox width/height ratio
        aspect_ratio_max (10.0)  Max bbox width/height ratio
        dilation_iter    (3)     Dilation iterations for merging
        kernel_w         (30)    Horizontal dilation kernel width
        kernel_h         (5)     Vertical dilation kernel height
        padding          (4)     Pixel padding added around each bbox
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.min_area = cfg.get("min_area", 50)
        self.max_area = cfg.get("max_area", 10000)
        self.ar_min = cfg.get("aspect_ratio_min", 0.1)
        self.ar_max = cfg.get("aspect_ratio_max", 10.0)
        self.dilation_iter = cfg.get("dilation_iter", 3)
        self.kernel_w = cfg.get("kernel_w", 30)
        self.kernel_h = cfg.get("kernel_h", 5)
        self.padding = cfg.get("padding", 4)

    def detect(self, image: np.ndarray) -> list[dict]:
        """Detect text regions in a BGR or grayscale image.

        Args:
            image: Input image (BGR or grayscale numpy array).

        Returns:
            List of region dicts sorted in reading order.
        """
        gray = self._to_gray(image)
        h_img, w_img = gray.shape

        # ── Preprocess for MSER ────────────────────────────────────────────
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        denoised = cv2.medianBlur(blurred, 3)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # ── MSER to find character blobs ───────────────────────────────────
        mser = cv2.MSER_create(5, self.min_area, self.max_area)
        regions, _ = mser.detectRegions(gray)

        # Draw character blobs onto a mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        for region in regions:
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            ar = w / max(h, 1)
            if self.ar_min <= ar <= self.ar_max:
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # ── Dilate to merge nearby characters into text lines ──────────────
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.kernel_w, self.kernel_h)
        )
        dilated = cv2.dilate(mask, kernel, iterations=self.dilation_iter)

        # ── Find contours of merged text blocks ───────────────────────────
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: list[dict] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Add padding, clip to image boundaries
            x = max(0, x - self.padding)
            y = max(0, y - self.padding)
            w = min(w_img - x, w + 2 * self.padding)
            h = min(h_img - y, h + 2 * self.padding)

            if w < 10 or h < 5:
                continue

            detections.append({
                "bbox": (x, y, w, h),
                "score": -1.0,  # MSER does not give a confidence score
                "type": "text_line",
            })

        # ── Sort in reading order: top-to-bottom, then left-to-right ──────
        detections.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
        return detections

    # ------------------------------------------------------------------
    # Fallback: simple gradient-based detector for very clean documents
    # ------------------------------------------------------------------

    def detect_simple(self, image: np.ndarray) -> list[dict]:
        """Simpler morphological detector — useful for clean printed documents.

        Uses Sobel gradient + close/open morphology to group text runs.

        Args:
            image: BGR or grayscale image.

        Returns:
            List of region dicts.
        """
        gray = self._to_gray(image)
        h_img, w_img = gray.shape

        # Gradient to highlight text edges
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(grad_x, grad_y)
        grad = cv2.convertScaleAbs(grad)

        _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_w, self.kernel_h))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 20 or h < 8:
                continue
            detections.append({"bbox": (x, y, w, h), "score": -1.0, "type": "text_line"})

        detections.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
        return detections

    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
