"""PaddleOCR adapter and structured result normalization."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import numpy as np

from api.settings import get_settings


class OcrEngineError(Exception):
    """The OCR runtime could not initialize or process a page."""


@dataclass(frozen=True)
class TextRegion:
    polygon: list[list[float]]
    text: str
    confidence: float
    # Per-word consensus details from the ensemble engine (text, support, sources, alternatives).
    words: list[dict] | None = None


class OcrEngine(Protocol):
    def recognize(self, image: np.ndarray) -> list[TextRegion]: ...


class PaddleOcrEngine:
    """CPU PaddleOCR adapter; model construction is intentionally lazy."""

    def __init__(self, language: str = "en", det_box_thresh: float = 0.5) -> None:
        try:
            from paddleocr import PaddleOCR

            # PP-OCR's default box threshold (0.6) drops faint handwritten lines;
            # 0.5 keeps them without adding noise on printed pages.
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=language,
                use_gpu=False,
                show_log=False,
                det_db_box_thresh=det_box_thresh,
            )
        except Exception as exc:
            raise OcrEngineError(
                "PaddleOCR could not be initialized. Check the model cache and runtime dependencies."
            ) from exc
        self._lock = threading.Lock()

    def recognize(self, image: np.ndarray) -> list[TextRegion]:
        from api import progress

        progress.report("recognize")  # PP-OCR detects and recognises in one call
        try:
            with self._lock:
                result = self._ocr.ocr(image, cls=True)
        except Exception as exc:
            raise OcrEngineError("PaddleOCR failed to process the page.") from exc
        return _normalize_result(result)

    def recognize_crops(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Recognition only, on pre-cut text-line images (used by the consensus engine)."""
        results: list[tuple[str, float]] = []
        try:
            with self._lock:
                for crop in crops:
                    bgr = crop if crop.ndim == 3 else np.stack([crop] * 3, axis=-1)
                    output = self._ocr.ocr(bgr, det=False, cls=False)
                    try:
                        text, score = output[0][0]
                    except (IndexError, TypeError, ValueError):
                        text, score = "", 0.0
                    results.append((" ".join(str(text).split()), float(score)))
        except Exception as exc:
            raise OcrEngineError("PaddleOCR recognition failed.") from exc
        return results

    def detect(self, image: np.ndarray) -> list[list[list[float]]]:
        """Text-line detection only (DB detector); returns quadrilateral polygons."""
        try:
            with self._lock:
                result = self._ocr.ocr(image, det=True, rec=False, cls=False)
        except Exception as exc:
            raise OcrEngineError("PaddleOCR text detection failed.") from exc
        boxes = result[0] if isinstance(result, list) and result and isinstance(result[0], list) else result
        polygons = []
        for box in boxes or []:
            try:
                polygon = [[float(point[0]), float(point[1])] for point in box]
            except (TypeError, ValueError, IndexError):
                continue
            if len(polygon) == 4 and all(math.isfinite(v) for point in polygon for v in point):
                polygons.append(polygon)
        return polygons


@lru_cache(maxsize=1)
def get_ocr_engine() -> PaddleOcrEngine:
    settings = get_settings()
    return PaddleOcrEngine(language=settings.language, det_box_thresh=settings.det_box_thresh)


def _normalize_result(result: Any) -> list[TextRegion]:
    if not result:
        return []
    lines = result[0] if isinstance(result, list) and len(result) == 1 else result
    if lines is None:
        return []
    if not isinstance(lines, list):
        raise OcrEngineError("PaddleOCR returned an unsupported result.")

    regions = []
    for line in lines:
        if not line or len(line) < 2:
            continue
        points, recognition = line[0], line[1]
        if not isinstance(recognition, (list, tuple)) or len(recognition) < 2:
            continue
        text, confidence = recognition[0], recognition[1]
        if not isinstance(text, str):
            continue
        try:
            polygon = [[float(point[0]), float(point[1])] for point in points]
            score = float(confidence)
        except (TypeError, ValueError, IndexError) as exc:
            raise OcrEngineError("PaddleOCR returned malformed region coordinates.") from exc
        if not math.isfinite(score) or any(
            not math.isfinite(coordinate) for point in polygon for coordinate in point
        ):
            raise OcrEngineError("PaddleOCR returned non-finite region data.")
        regions.append(TextRegion(polygon=polygon, text=text, confidence=score))
    return regions
