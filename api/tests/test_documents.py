import io

import fitz
import numpy as np
import pytest
from PIL import Image

from api.documents import DocumentError, decode_document
from api.settings import Settings


@pytest.fixture
def settings():
    return Settings(max_upload_bytes=100_000, max_pages=2, max_image_dimension=500,
                    max_pixels_per_page=100_000, pdf_render_scale=2)


def image_bytes(size=(20, 10), format="PNG"):
    stream = io.BytesIO()
    Image.new("RGB", size, "white").save(stream, format=format)
    return stream.getvalue()


def test_decodes_supported_image_as_bgr(settings):
    pages = decode_document(image_bytes(), settings)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].image.shape == (10, 20, 3)


def test_rejects_unsupported_and_corrupt_files(settings):
    with pytest.raises(DocumentError, match="Unsupported file type"):
        decode_document(b"plain text", settings)
    with pytest.raises(DocumentError, match="invalid"):
        decode_document(b"\x89PNG\r\n\x1a\nbroken", settings)


def test_rejects_dimensions_and_page_count_over_limits(settings):
    with pytest.raises(DocumentError, match="dimensions"):
        decode_document(image_bytes(size=(501, 10)), settings)

    document = fitz.open()
    document.new_page()
    document.new_page()
    document.new_page()
    pdf = document.tobytes()
    document.close()
    with pytest.raises(DocumentError, match="pages"):
        decode_document(pdf, settings)


def test_rejects_aggregate_pixels_before_decoding_pages(settings):
    bounded = Settings(
        max_upload_bytes=settings.max_upload_bytes,
        max_pages=settings.max_pages,
        max_image_dimension=settings.max_image_dimension,
        max_pixels_per_page=settings.max_pixels_per_page,
        max_total_pixels=199,
        pdf_render_scale=settings.pdf_render_scale,
    )

    frames = [Image.new("RGB", (10, 10), "white") for _ in range(2)]
    tiff = io.BytesIO()
    frames[0].save(tiff, format="TIFF", save_all=True, append_images=frames[1:])
    with pytest.raises(DocumentError, match="total decoded-pixel limit"):
        decode_document(tiff.getvalue(), bounded)

    document = fitz.open()
    document.new_page(width=10, height=10)
    document.new_page(width=10, height=10)
    pdf = document.tobytes()
    document.close()
    with pytest.raises(DocumentError, match="total decoded-pixel limit"):
        decode_document(pdf, bounded)


def test_renders_pdf_pages_with_bounded_dimensions(settings):
    document = fitz.open()
    document.new_page(width=100, height=50)
    pdf = document.tobytes()
    document.close()
    pages = decode_document(pdf, settings)
    assert len(pages) == 1
    assert pages[0].image.shape[:2] == (100, 200)
    assert pages[0].image.dtype == np.uint8


def test_rejects_empty_upload(settings):
    with pytest.raises(DocumentError, match="empty"):
        decode_document(b"", settings)
