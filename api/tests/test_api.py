import io

import fitz
import numpy as np
import cv2
from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app
from api.ocr_engine import OcrEngineError, TextRegion
from api.settings import Settings


class FakeEngine:
    def recognize(self, image):
        return [TextRegion(
            polygon=[[1.0, 2.0], [20.0, 2.0], [20.0, 12.0], [1.0, 12.0]],
            text="hello OCR",
            confidence=0.95,
        )]


def png_bytes():
    image = io.BytesIO()
    page = Image.new("RGB", (640, 320), "white")
    cv2_image = np.asarray(page).copy()
    cv2.putText(cv2_image, "hello OCR", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    Image.fromarray(cv2_image).save(image, format="PNG")
    return image.getvalue()


def client(engine_provider=None, max_upload_bytes=20_000, max_pages=2):
    app = create_app(
        settings=Settings(max_upload_bytes=max_upload_bytes, max_pages=max_pages),
        engine_provider=engine_provider or FakeEngine,
    )
    return TestClient(app)


def test_default_upload_size_is_100_mib():
    assert Settings().max_upload_bytes == 100 * 1024 * 1024
    assert Settings().max_pages == 100


def test_health_and_readiness():
    test_client = client()
    assert test_client.get("/api/v1/health/live").json() == {"status": "ok"}
    assert test_client.get("/api/v1/health/ready").json() == {"status": "ready"}


def test_recognizes_image_and_sanitizes_filename():
    response = client().post(
        "/api/v1/ocr",
        files={"file": ("../../private\\note.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["filename"] == "note.png"
    assert result["model"] == "PaddleOCR"
    assert result["language"] == "en"
    page = result["pages"][0]
    assert page["text"] == "hello OCR"
    assert page["regions"][0]["confidence"] == 0.95
    assert page["blocks"][0]["markdown"] == "hello OCR"
    assert page["blocks"][0]["kind"] == "text"
    assert page["preview_image"]
    assert result["markdown"].endswith("hello OCR")


def test_renders_and_recognizes_multiple_pdf_pages():
    document = fitz.open()
    document.new_page()
    document.new_page()
    pdf = document.tobytes()
    document.close()
    response = client().post(
        "/api/v1/ocr",
        files={"file": ("scan.pdf", pdf, "application/pdf")},
    )
    assert response.status_code == 200
    pages = response.json()["pages"]
    assert [page["page_number"] for page in pages] == [1, 2]
    assert all(page["preview_image"] for page in pages)


def test_upload_size_page_count_and_type_limits():
    too_large = client(max_upload_bytes=10).post(
        "/api/v1/ocr",
        files={"file": ("scan.png", png_bytes(), "image/png")},
    )
    assert too_large.status_code == 413

    document = fitz.open()
    document.new_page()
    document.new_page()
    document.new_page()
    pdf = document.tobytes()
    document.close()
    too_many_pages = client().post(
        "/api/v1/ocr",
        files={"file": ("scan.pdf", pdf, "application/pdf")},
    )
    assert too_many_pages.status_code == 413

    unsupported = client().post(
        "/api/v1/ocr",
        files={"file": ("note.txt", b"not an image", "text/plain")},
    )
    assert unsupported.status_code == 415


def test_accepts_100_page_pdf_and_rejects_101_pages():
    test_client = client(max_pages=100)

    def make_pdf(page_count):
        document = fitz.open()
        for _ in range(page_count):
            document.new_page(width=10, height=10)
        pdf = document.tobytes()
        document.close()
        return pdf

    accepted = test_client.post(
        "/api/v1/ocr",
        files={"file": ("hundred-pages.pdf", make_pdf(100), "application/pdf")},
    )
    assert accepted.status_code == 200
    assert len(accepted.json()["pages"]) == 100

    rejected = test_client.post(
        "/api/v1/ocr",
        files={"file": ("hundred-one-pages.pdf", make_pdf(101), "application/pdf")},
    )
    assert rejected.status_code == 413


def test_engine_failure_is_not_reported_as_success():
    def broken_engine():
        raise OcrEngineError("internal engine failure")

    test_client = client(engine_provider=broken_engine)
    response = test_client.post(
        "/api/v1/ocr",
        files={"file": ("scan.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == (
        "OCR processing failed. Check the server's model cache and logs."
    )
    assert test_client.get("/api/v1/health/ready").status_code == 503
