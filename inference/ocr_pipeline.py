"""
End-to-End OCR Inference Pipeline for DeepOCR (research / Streamlit / CLI path).

Usage:
    pipeline = OCRPipeline.from_config(model_name="crnn")
    result = pipeline.run(image)
    print(result["text"])

The pipeline:
  1. Preprocess image (deskew / denoise / contrast, for display and detection).
  2. Detect text regions (PP-OCR DBNet when available, else OpenCV MSER).
  3. Analyse layout (title / heading / paragraph, columns, reading order).
  4. Recognise each region with a trained model (CNN-BiLSTM, CRNN) or TrOCR.
  5. Post-process recognised text (character confusions + spelling).
  6. Return a structured result.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from datasets.ocr_dataset import build_vocabulary
from detection.opencv_detector import OpenCVTextDetector
from layout.layout_analyzer import LayoutAnalyzer
from models.recognizer import LineRecognizer
from postprocessing.text_corrector import TextCorrector
from preprocessing.image_processor import ImageProcessor
from utils.config_loader import get_project_root, load_all_configs
from utils.logger import get_logger

logger = get_logger(__name__)

_CTC_MODELS = {"cnn_bilstm", "crnn"}
_TRANSFORMER_ALIASES = {"transformer_ocr", "trocr"}


class PaddleLineDetector:
    """Adapter exposing PP-OCR's DBNet text detector with the detector interface."""

    def __init__(self) -> None:
        from api.ocr_engine import get_ocr_engine

        self._engine = get_ocr_engine()

    def detect(self, image: np.ndarray) -> list[dict]:
        detections = []
        for polygon in self._engine.detect(image):
            pts = np.asarray(polygon, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            pad = max(2, int(h * 0.12))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1 = min(image.shape[1], x + w + pad)
            y1 = min(image.shape[0], y + h + pad)
            detections.append({"bbox": (x0, y0, x1 - x0, y1 - y0), "score": 1.0, "type": "text_line",
                               "polygon": polygon})
        detections.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
        return detections


def default_detector():
    """PP-OCR DBNet if PaddleOCR is installed, otherwise the OpenCV MSER detector."""
    try:
        return PaddleLineDetector()
    except Exception as exc:  # noqa: BLE001 - optional dependency
        logger.info(f"PaddleOCR detector unavailable ({exc}); using OpenCV MSER detector.")
        return OpenCVTextDetector({"max_area": 200_000})


class OCRPipeline:
    """Full OCR pipeline: preprocess -> detect -> layout -> recognise -> post-process.

    Args:
        model: Optional PyTorch CTC model (used when ``recognizer`` is not given).
        vocab: (chars, char2idx, idx2char) vocabulary for ``model``.
        model_type: "ctc" (only CTC models can be wrapped from a bare model).
        preprocessor: ImageProcessor instance.
        detector: Object with ``detect(image) -> [{"bbox": (x, y, w, h), ...}]``.
        layout_analyzer: LayoutAnalyzer instance.
        corrector: TextCorrector instance (``None`` to use the default).
        device: Torch device.
        recognizer: Object with ``recognize(list[np.ndarray]) -> [(text, conf)]``.
    """

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        vocab: tuple | None = None,
        model_type: str = "ctc",
        preprocessor: ImageProcessor | None = None,
        detector=None,
        layout_analyzer: LayoutAnalyzer | None = None,
        corrector: TextCorrector | None = None,
        device: torch.device | str = "auto",
        recognizer=None,
        deskew: bool = True,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if recognizer is None:
            if model is None or vocab is None:
                raise ValueError("Provide either a recognizer or a model with its vocabulary.")
            if model_type != "ctc":
                raise ValueError("Bare models are only supported for CTC architectures; pass a recognizer.")
            name = "crnn" if model.__class__.__name__ == "CRNN" else "cnn_bilstm"
            meta = {
                "model_name": name,
                "charset": "".join(vocab[0][1:]),
                "image_height": getattr(model, "image_height", 64),
                "image_width": 512,
            }
            recognizer = LineRecognizer(model, meta, device)
        self.recognizer = recognizer
        self.preprocessor = preprocessor or ImageProcessor({"resize": False})
        self.detector = detector or OpenCVTextDetector({"max_area": 200_000})
        self.layout = layout_analyzer or LayoutAnalyzer()
        self.corrector = corrector or TextCorrector()
        self.checkpoint_loaded = False
        self.model_name = getattr(recognizer, "model_name", "custom")
        self.deskew = deskew

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        model_name: str = "cnn_bilstm",
        checkpoint_path: str | Path | None = None,
        detector: str | object = "auto",
    ) -> "OCRPipeline":
        """Build a pipeline around a trained checkpoint (or pretrained TrOCR).

        Raises:
            FileNotFoundError: When a CTC model has no trained checkpoint.
            RuntimeError: When the checkpoint exists but cannot be loaded.
        """
        cfg = load_all_configs()
        root = get_project_root()
        if model_name in _TRANSFORMER_ALIASES:
            from models.recognizer import TrOCRRecognizer

            encoder = cfg.get("transformer_ocr", {}).get("encoder", "microsoft/trocr-small-handwritten")
            recognizer = TrOCRRecognizer(encoder)
        elif model_name in _CTC_MODELS:
            path = Path(checkpoint_path) if checkpoint_path else root / "checkpoints" / model_name / "best_model.pth"
            if not path.exists():
                raise FileNotFoundError(
                    f"No trained checkpoint found at {path}. "
                    f"Train the model first: python scripts/train.py --model {model_name}"
                )
            try:
                recognizer = LineRecognizer.from_checkpoint(path)
            except Exception as exc:
                raise RuntimeError(f"Failed to load checkpoint {path}: {exc}") from exc
        else:
            raise ValueError(f"Unknown model: {model_name}. Choose cnn_bilstm, crnn or transformer_ocr.")

        det = default_detector() if detector == "auto" else (OpenCVTextDetector() if detector == "opencv" else detector)
        pipeline = cls(recognizer=recognizer, detector=det)
        pipeline.checkpoint_loaded = True
        pipeline.model_name = model_name
        return pipeline

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        image: np.ndarray | Image.Image | str | Path,
        return_regions: bool = True,
        correct: bool = True,
    ) -> dict[str, Any]:
        """Run the complete OCR pipeline on an image.

        Returns:
            Dict with keys: text, regions, layout_blocks, preprocessed,
            bbox_image, inference_ms, checkpoint_loaded, timings.
        """
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        original_bgr = self._load_image(image)

        mark = time.perf_counter()
        # Straighten the page first so detection and line crops see horizontal text.
        deskew_angle = 0.0
        if self.deskew:
            from api.preprocess import estimate_skew, rotate_page

            deskew_angle = estimate_skew(original_bgr)
            if abs(deskew_angle) >= 0.3:
                original_bgr = rotate_page(original_bgr, deskew_angle)
        h_img, w_img = original_bgr.shape[:2]
        preprocessed = self.preprocessor.process(original_bgr)
        timings["preprocess_ms"] = (time.perf_counter() - mark) * 1000

        mark = time.perf_counter()
        detections = self.detector.detect(original_bgr)
        if not detections:
            detections = [{"bbox": (0, 0, w_img, h_img), "score": -1.0, "type": "text_line"}]
        timings["detect_ms"] = (time.perf_counter() - mark) * 1000

        mark = time.perf_counter()
        layout_blocks = self.layout.analyze(detections, image_width=w_img, image_height=h_img)
        polygons = {tuple(d["bbox"]): d.get("polygon") for d in detections}
        timings["layout_ms"] = (time.perf_counter() - mark) * 1000

        mark = time.perf_counter()
        crops, kept = [], []
        for block in layout_blocks:
            x, y, bw, bh = block.bbox
            polygon = polygons.get(tuple(block.bbox))
            if polygon is not None:
                from api.engines import crop_polygon

                crop = crop_polygon(original_bgr, polygon)  # rectified along the text line
            else:
                crop = original_bgr[y: y + bh, x: x + bw]
            if crop.size:
                crops.append(crop)
                kept.append(block)
            else:
                block.text = ""
        outputs = self.recognizer.recognize(crops) if crops else []
        timings["recognize_ms"] = (time.perf_counter() - mark) * 1000

        mark = time.perf_counter()
        region_results = []
        for block, (raw_text, confidence) in zip(kept, outputs):
            block.text = self.corrector.correct(raw_text) if correct else raw_text
            block.score = confidence
            if return_regions:
                region_results.append({
                    "bbox": block.bbox,
                    "region_type": block.region_type,
                    "reading_order": block.reading_order,
                    "text": block.text,
                    "raw_text": raw_text,
                    "confidence": round(float(confidence), 4),
                })
        timings["postprocess_ms"] = (time.perf_counter() - mark) * 1000

        full_text = self.layout.to_text(layout_blocks, separator="\n")

        from utils.visualization import draw_bboxes
        bbox_img = draw_bboxes(
            original_bgr,
            [b.bbox for b in layout_blocks],
            labels=[b.region_type for b in layout_blocks],
        )

        return {
            "text": full_text,
            "regions": region_results,
            "layout_blocks": layout_blocks,
            "preprocessed": preprocessed,
            "bbox_image": bbox_img,
            "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
            "timings": {key: round(value, 1) for key, value in timings.items()},
            "deskew_angle": deskew_angle,
            "checkpoint_loaded": self.checkpoint_loaded,
        }

    @staticmethod
    def _load_image(image) -> np.ndarray:
        """Load image from various input types into BGR numpy array."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not read image: {image}")
            return img
        if isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image
        raise TypeError(f"Unsupported image type: {type(image)}")


__all__ = ["OCRPipeline", "PaddleLineDetector", "default_detector", "build_vocabulary"]
