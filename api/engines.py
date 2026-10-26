"""OCR engine registry: pretrained PaddleOCR and DeepOCR's own trained recognisers.

Every engine exposes ``recognize(image_bgr) -> list[TextRegion]``.

* ``paddle``      — PaddleOCR detection + recognition (pretrained reference).
* ``crnn``        — PaddleOCR DB text-line detection + DeepOCR CRNN (trained here).
* ``cnn_bilstm``  — PaddleOCR DB text-line detection + DeepOCR CNN-BiLSTM baseline.
* ``trocr``       — PaddleOCR DB text-line detection + TrOCR Transformer (transfer learning).

The detect-then-recognise engines fall back to the OpenCV MSER detector from
``detection/`` when PaddleOCR is unavailable, so the custom models keep working
on machines where Paddle cannot be installed.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from api.ocr_engine import OcrEngine, OcrEngineError, TextRegion, get_ocr_engine
from api.settings import Settings

logger = logging.getLogger("deepocr.engines")


@dataclass(frozen=True)
class EngineSpec:
    id: str
    label: str
    family: str
    description: str
    detector: str
    recognizer: str


CATALOG: dict[str, EngineSpec] = {
    spec.id: spec
    for spec in [
        EngineSpec("paddle", "PaddleOCR", "Pretrained reference",
                   "Industrial PP-OCR pipeline. Strong on clean printed text.",
                   "DBNet (PP-OCR)", "PP-OCR English recogniser"),
        EngineSpec("crnn", "DeepOCR · CRNN", "Ours · trained from scratch",
                   "VGG-style CNN + 2×BiLSTM + CTC (Shi et al.), trained on IAM handwriting + synthetic print.",
                   "DBNet (PP-OCR)", "CRNN + CTC"),
        EngineSpec("cnn_bilstm", "DeepOCR · CNN-BiLSTM", "Ours · baseline",
                   "4-block CNN feature extractor + stacked BiLSTM + CTC. The project baseline.",
                   "DBNet (PP-OCR)", "CNN-BiLSTM + CTC"),
        EngineSpec("trocr", "TrOCR Transformer", "Advanced · transfer learning",
                   "Vision Transformer encoder + autoregressive Transformer decoder, pretrained on handwriting.",
                   "DBNet (PP-OCR)", "TrOCR-small (ViT + Transformer)"),
        EngineSpec("trocr_ft", "TrOCR · fine-tuned", "Advanced · fine-tuned by us",
                   "The TrOCR Transformer further fine-tuned on our IAM + synthetic training data.",
                   "DBNet (PP-OCR)", "TrOCR-small fine-tuned"),
    ]
}


def crop_polygon(image: np.ndarray, polygon: list[list[float]], pad_ratio: float = 0.15) -> np.ndarray:
    """Perspective-rectify a quadrilateral text region with a small context margin."""
    pts = np.asarray(polygon, dtype=np.float32)
    if pts.shape != (4, 2):
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        pts = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    width = float(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[3] - pts[2]), 1))
    height = float(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]), 1))
    pad = height * pad_ratio
    dst = np.float32([[pad, pad], [pad + width, pad], [pad + width, pad + height], [pad, pad + height]])
    matrix = cv2.getPerspectiveTransform(pts, dst)
    size = (int(round(width + 2 * pad)), int(round(height + 2 * pad)))
    crop = cv2.warpPerspective(image, matrix, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if crop.shape[0] > crop.shape[1] * 1.5:  # vertical text line
        crop = np.rot90(crop)
    return np.ascontiguousarray(crop)


class ContourDetector:
    """Fallback line detector using the research OpenCV MSER detector."""

    def __init__(self) -> None:
        from detection.opencv_detector import OpenCVTextDetector

        self._detector = OpenCVTextDetector({"max_area": 200_000})

    def detect(self, image: np.ndarray) -> list[list[list[float]]]:
        polygons = []
        for item in self._detector.detect(image):
            x, y, w, h = item["bbox"]
            polygons.append([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        return polygons


class DetectRecognizeEngine:
    """Two-stage engine: text-line detection followed by a line recogniser."""

    def __init__(self, detector_provider: Callable[[], object], recognizer, name: str) -> None:
        self._detector_provider = detector_provider
        self.recognizer = recognizer
        self.name = name

    def recognize(self, image: np.ndarray) -> list[TextRegion]:
        detector = self._detector_provider()
        polygons = detector.detect(image)
        if not polygons:
            return []
        crops = [crop_polygon(image, polygon) for polygon in polygons]
        try:
            results = self.recognizer.recognize(crops)
        except Exception as exc:  # pragma: no cover - model runtime failure
            raise OcrEngineError(f"{self.name} recognition failed.") from exc
        return [
            TextRegion(polygon=polygon, text=text, confidence=float(max(0.0, min(1.0, confidence))))
            for polygon, (text, confidence) in zip(polygons, results)
            if text.strip()
        ]


class EngineRegistry:
    """Lazily constructs engines, caches them, and reports their status."""

    def __init__(self, settings: Settings, overrides: dict[str, Callable[[], OcrEngine]] | None = None) -> None:
        self.settings = settings
        self._providers: dict[str, Callable[[], OcrEngine]] = {}
        self._instances: dict[str, OcrEngine] = {}
        self._errors: dict[str, str] = {}
        self._load_ms: dict[str, float] = {}
        self._mtimes: dict[str, float] = {}
        # Re-entrant: building a detect+recognise engine may resolve the Paddle detector.
        self._lock = threading.RLock()
        if overrides is not None:
            self._providers.update(overrides)
        else:
            self._providers.update({
                "paddle": get_ocr_engine,
                "crnn": lambda: self._ctc_engine("crnn"),
                "cnn_bilstm": lambda: self._ctc_engine("cnn_bilstm"),
                "trocr": self._trocr_engine,
                "trocr_ft": lambda: self._trocr_engine(str(self.finetuned_trocr_dir()), "trocr_ft"),
            })

    # -- construction --------------------------------------------------
    def _detector(self):
        paddle = self._providers.get("paddle")
        if paddle is not None:
            try:
                engine = self.get("paddle")
                if hasattr(engine, "detect"):
                    return engine
            except OcrEngineError:
                logger.warning("PaddleOCR unavailable; using the OpenCV contour detector.")
        if "contour" not in self._instances:
            self._instances["contour"] = ContourDetector()
        return self._instances["contour"]

    def checkpoint_path(self, model_name: str) -> Path:
        return self.settings.checkpoints_dir / model_name / "best_model.pth"

    def _ctc_engine(self, model_name: str) -> DetectRecognizeEngine:
        from models.recognizer import LineRecognizer

        path = self.checkpoint_path(model_name)
        if not path.is_file():
            raise OcrEngineError(
                f"No trained {model_name} checkpoint. Run: python scripts/train.py --model {model_name}"
            )
        try:
            recognizer = LineRecognizer.from_checkpoint(path)
        except Exception as exc:
            raise OcrEngineError(f"Could not load the {model_name} checkpoint: {exc}") from exc
        return DetectRecognizeEngine(self._detector, recognizer, model_name)

    def finetuned_trocr_dir(self) -> Path:
        return self.settings.checkpoints_dir / "trocr_finetuned"

    def _trocr_engine(self, model: str | None = None, name: str = "trocr") -> DetectRecognizeEngine:
        model = model or self.settings.trocr_model
        try:
            from models.recognizer import TrOCRRecognizer

            recognizer = TrOCRRecognizer(model)
        except Exception as exc:
            raise OcrEngineError(f"TrOCR could not be loaded ({model}): {exc}") from exc
        return DetectRecognizeEngine(self._detector, recognizer, name)

    # -- access ----------------------------------------------------------
    def ids(self) -> list[str]:
        return list(self._providers)

    def _checkpoint_mtime(self, engine_id: str) -> float | None:
        if engine_id not in ("crnn", "cnn_bilstm"):
            return None
        try:
            return self.checkpoint_path(engine_id).stat().st_mtime
        except OSError:
            return None

    def _is_stale(self, engine_id: str) -> bool:
        """True when training has written a newer checkpoint since the engine was loaded."""
        loaded = self._mtimes.get(engine_id)
        current = self._checkpoint_mtime(engine_id)
        return loaded is not None and current is not None and current > loaded

    def get(self, engine_id: str) -> OcrEngine:
        if engine_id not in self._providers:
            raise KeyError(engine_id)
        if engine_id in self._instances and not self._is_stale(engine_id):
            return self._instances[engine_id]
        with self._lock:
            if engine_id in self._instances and not self._is_stale(engine_id):
                return self._instances[engine_id]
            mtime = self._checkpoint_mtime(engine_id)
            if mtime is not None and time.time() - mtime < 3:
                time.sleep(3)  # let an in-progress torch.save finish writing
            started = time.perf_counter()
            try:
                engine = self._providers[engine_id]()
            except OcrEngineError as exc:
                self._errors[engine_id] = str(exc)
                raise
            except Exception as exc:
                self._errors[engine_id] = str(exc)
                raise OcrEngineError(str(exc)) from exc
            self._instances[engine_id] = engine
            if mtime is not None:
                self._mtimes[engine_id] = mtime
            self._errors.pop(engine_id, None)
            self._load_ms[engine_id] = round((time.perf_counter() - started) * 1000, 1)
            return engine

    def warmup(self, engine_ids: list[str]) -> None:
        for engine_id in engine_ids:
            if engine_id in self._providers and self.is_installable(engine_id):
                try:
                    self.get(engine_id)
                    logger.info("Engine %s ready.", engine_id)
                except OcrEngineError as exc:
                    logger.warning("Engine %s unavailable: %s", engine_id, exc)

    def is_installable(self, engine_id: str) -> bool:
        """Cheap availability check that does not load models."""
        if engine_id in ("crnn", "cnn_bilstm") and self._providers.get(engine_id) is not None:
            return self.checkpoint_path(engine_id).is_file() or engine_id in self._instances
        if engine_id == "trocr_ft":
            return (self.finetuned_trocr_dir() / "config.json").is_file() or engine_id in self._instances
        return True

    # -- reporting ---------------------------------------------------------
    def _evaluation_summary(self) -> dict:
        path = self.settings.results_dir / "evaluation" / "summary.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def status(self) -> list[dict]:
        summary = self._evaluation_summary().get("models", {})
        rows = []
        for engine_id in self.ids():
            spec = CATALOG.get(engine_id) or EngineSpec(engine_id, engine_id, "Custom", "", "", "")
            instance = self._instances.get(engine_id)
            info = {}
            recognizer = getattr(instance, "recognizer", None)
            if recognizer is not None and hasattr(recognizer, "describe"):
                info = recognizer.describe()
            if engine_id in ("crnn", "cnn_bilstm") and not info:
                info = self._checkpoint_info(engine_id)
            rows.append({
                "id": engine_id,
                "label": spec.label,
                "family": spec.family,
                "description": spec.description,
                "detector": spec.detector,
                "recognizer": spec.recognizer,
                "available": self.is_installable(engine_id)
                and (engine_id in ("crnn", "cnn_bilstm") or engine_id not in self._errors),
                "loaded": instance is not None,
                "load_ms": self._load_ms.get(engine_id),
                "error": self._errors.get(engine_id),
                "info": info,
                "metrics": summary.get(engine_id),
            })
        return rows

    def _checkpoint_info(self, model_name: str) -> dict:
        history = self.settings.results_dir / "training" / f"{model_name}_history.json"
        try:
            payload = json.loads(history.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        meta = payload.get("meta") or {}
        return {
            "architecture": model_name,
            "parameters": meta.get("parameters"),
            "input": f"{meta.get('image_height')}x{meta.get('image_width')}",
            "epoch": payload.get("epoch"),
            "val_cer": payload.get("best_val_cer"),
            "datasets": meta.get("datasets"),
            "training": not payload.get("finished", False),
        }
