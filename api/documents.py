"""Bounded, in-memory decoding for supported image and PDF uploads."""
from __future__ import annotations

import io
import math
import warnings
from dataclasses import dataclass

import fitz
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from api.settings import Settings

SUPPORTED_IMAGE_FORMATS = {"BMP", "JPEG", "PNG", "TIFF", "WEBP"}
SUPPORTED_IMAGE_SIGNATURES = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"BM",
    b"II*\x00",
    b"MM\x00*",
    b"II+\x00",
    b"MM\x00+",
)


class DocumentError(Exception):
    """An upload was invalid or exceeded a configured resource limit."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class PageImage:
    page_number: int
    image: np.ndarray


def decode_document(content: bytes, settings: Settings) -> list[PageImage]:
    if not content:
        raise DocumentError("The uploaded file is empty.")

    if content.startswith(b"%PDF-"):
        return _decode_pdf(content, settings)
    return _decode_image(content, settings)


def _decode_pdf(content: bytes, settings: Settings) -> list[PageImage]:
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except (fitz.FileDataError, ValueError) as exc:
        raise DocumentError("The PDF file is invalid or unsupported.") from exc

    try:
        if document.needs_pass:
            raise DocumentError("Encrypted PDFs are not supported.", 422)
        if len(document) == 0:
            raise DocumentError("The PDF contains no pages.")
        if len(document) > settings.max_pages:
            raise DocumentError(
                f"The PDF has {len(document)} pages; the limit is {settings.max_pages}.",
                413,
            )

        page_scales = []
        total_pixels = 0
        for index, page in enumerate(document):
            rect = page.rect
            width, height = float(rect.width), float(rect.height)
            if width <= 0 or height <= 0:
                raise DocumentError(f"PDF page {index + 1} has invalid dimensions.")
            scale = min(
                settings.pdf_render_scale,
                settings.max_image_dimension / max(width, height),
                math.sqrt(settings.max_pixels_per_page / (width * height)),
            )
            output_pixels = math.ceil(width * scale) * math.ceil(height * scale)
            total_pixels += output_pixels
            if total_pixels > settings.max_total_pixels:
                raise DocumentError(
                    "The PDF exceeds the total decoded-pixel limit of "
                    f"{settings.max_total_pixels:,} pixels.",
                    413,
                )
            page_scales.append(scale)

        pages: list[PageImage] = []
        for index, (page, scale) in enumerate(zip(document, page_scales)):
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                colorspace=fitz.csRGB,
                alpha=False,
            )
            if pixmap.width * pixmap.height > settings.max_pixels_per_page:
                raise DocumentError(f"PDF page {index + 1} exceeds the pixel limit.", 413)
            rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, 3
            )
            pages.append(PageImage(index + 1, np.ascontiguousarray(rgb[:, :, ::-1])))
        return pages
    except DocumentError:
        raise
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise DocumentError("The PDF could not be decoded safely.", 422) from exc
    finally:
        document.close()


def _decode_image(content: bytes, settings: Settings) -> list[PageImage]:
    if not any(content.startswith(signature) for signature in SUPPORTED_IMAGE_SIGNATURES) and not (
        content.startswith(b"RIFF") and content[8:12] == b"WEBP"
    ):
        raise DocumentError("Unsupported file type. Upload a PDF or a supported image.", 415)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(content))
            if image.format not in SUPPORTED_IMAGE_FORMATS:
                raise DocumentError(
                    "Unsupported file type. Upload a PDF or a supported image.", 415
                )
            frame_count = getattr(image, "n_frames", 1)
            if frame_count > settings.max_pages:
                raise DocumentError(
                    f"The image has {frame_count} frames; the limit is {settings.max_pages}.",
                    413,
                )

            frame_dimensions = []
            total_pixels = 0
            for index in range(frame_count):
                image.seek(index)
                _validate_image_size(image.width, image.height, settings)
                total_pixels += image.width * image.height
                if total_pixels > settings.max_total_pixels:
                    raise DocumentError(
                        "The image exceeds the total decoded-pixel limit of "
                        f"{settings.max_total_pixels:,} pixels.",
                        413,
                    )
                frame_dimensions.append((image.width, image.height))

            pages = []
            for index, dimensions in enumerate(frame_dimensions):
                image.seek(index)
                if (image.width, image.height) != dimensions:
                    raise DocumentError("The image dimensions changed during decoding.", 422)
                frame = ImageOps.exif_transpose(image.copy()).convert("RGB")
                rgb = np.asarray(frame, dtype=np.uint8)
                pages.append(PageImage(index + 1, np.ascontiguousarray(rgb[:, :, ::-1])))
            return pages
    except DocumentError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        ValueError,
    ) as exc:
        raise DocumentError(
            "The image is invalid, corrupted, or exceeds safe decoding limits."
        ) from exc


def _validate_image_size(width: int, height: int, settings: Settings) -> None:
    if width < 1 or height < 1:
        raise DocumentError("The image has invalid dimensions.")
    if max(width, height) > settings.max_image_dimension:
        raise DocumentError(
            f"Image dimensions exceed the {settings.max_image_dimension}px limit.",
            413,
        )
    if width * height > settings.max_pixels_per_page:
        raise DocumentError("The image exceeds the configured pixel limit.", 413)
