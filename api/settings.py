"""Environment-backed resource limits and paths for the web service."""
from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    max_upload_bytes: int = 100 * 1024 * 1024
    max_pages: int = 100
    max_image_dimension: int = 10_000
    max_pixels_per_page: int = 40_000_000
    max_total_pixels: int = 100_000_000
    pdf_render_scale: float = 2.0
    review_confidence_threshold: float = 0.45
    det_box_thresh: float = 0.5
    language: str = "en"
    default_engine: str = "paddle"
    trocr_model: str = "microsoft/trocr-small-handwritten"
    checkpoints_dir: Path = field(default=PROJECT_ROOT / "checkpoints")
    results_dir: Path = field(default=PROJECT_ROOT / "results")
    samples_dir: Path = field(default=PROJECT_ROOT / "samples")
    web_dist_dir: Path = field(default=PROJECT_ROOT / "web" / "dist")
    warmup: bool = False

    @classmethod
    def from_environment(cls) -> "Settings":
        settings = cls(
            max_upload_bytes=int(os.getenv("OCR_MAX_UPLOAD_BYTES", 100 * 1024 * 1024)),
            max_pages=int(os.getenv("OCR_MAX_PAGES", 100)),
            max_image_dimension=int(os.getenv("OCR_MAX_IMAGE_DIMENSION", 10_000)),
            max_pixels_per_page=int(os.getenv("OCR_MAX_PIXELS_PER_PAGE", 40_000_000)),
            max_total_pixels=int(os.getenv("OCR_MAX_TOTAL_PIXELS", 100_000_000)),
            pdf_render_scale=float(os.getenv("OCR_PDF_RENDER_SCALE", "2")),
            review_confidence_threshold=float(
                os.getenv("OCR_REVIEW_CONFIDENCE_THRESHOLD", "0.45")
            ),
            det_box_thresh=float(os.getenv("OCR_DET_BOX_THRESH", "0.5")),
            language=os.getenv("OCR_LANGUAGE", "en").strip(),
            default_engine=os.getenv("OCR_DEFAULT_ENGINE", "paddle").strip(),
            trocr_model=os.getenv("OCR_TROCR_MODEL", "microsoft/trocr-small-handwritten").strip(),
            checkpoints_dir=Path(os.getenv("OCR_CHECKPOINTS_DIR", PROJECT_ROOT / "checkpoints")),
            results_dir=Path(os.getenv("OCR_RESULTS_DIR", PROJECT_ROOT / "results")),
            samples_dir=Path(os.getenv("OCR_SAMPLES_DIR", PROJECT_ROOT / "samples")),
            web_dist_dir=Path(os.getenv("OCR_WEB_DIST", PROJECT_ROOT / "web" / "dist")),
            warmup=os.getenv("OCR_WARMUP", "1").strip().lower() in {"1", "true", "yes"},
        )
        if min(
            settings.max_upload_bytes,
            settings.max_pages,
            settings.max_image_dimension,
            settings.max_pixels_per_page,
            settings.max_total_pixels,
        ) < 1 or not math.isfinite(settings.pdf_render_scale) or settings.pdf_render_scale <= 0:
            raise ValueError("OCR resource limits and PDF render scale must be positive.")
        if not math.isfinite(settings.det_box_thresh) or not 0 < settings.det_box_thresh < 1:
            raise ValueError("OCR_DET_BOX_THRESH must be between 0 and 1.")
        if not settings.language.strip():
            raise ValueError("OCR_LANGUAGE must not be empty.")
        if (
            not math.isfinite(settings.review_confidence_threshold)
            or not 0 <= settings.review_confidence_threshold <= 1
        ):
            raise ValueError("OCR_REVIEW_CONFIDENCE_THRESHOLD must be between 0 and 1.")
        return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_environment()
