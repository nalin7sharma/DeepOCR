import io
import json

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from api.engines import DetectRecognizeEngine, crop_polygon
from api.main import create_app
from api.ocr_engine import TextRegion
from api.preprocess import estimate_skew, preprocess_page, rotate_page
from api.settings import Settings
from postprocessing.text_corrector import TextCorrector


class FakePaddle:
    def recognize(self, image):
        return [TextRegion([[10, 10], [200, 10], [200, 40], [10, 40]], "Teh invoice", 0.9)]

    def detect(self, image):
        return [[[10.0, 10.0], [200.0, 10.0], [200.0, 40.0], [10.0, 40.0]]]


class FakeLineRecognizer:
    def recognize(self, crops):
        return [("custom model text", 0.8) for _ in crops]


def png_bytes(angle: float = 0.0):
    page = np.full((400, 700, 3), 255, np.uint8)
    for row in range(5):
        cv2.putText(page, "DeepOCR line of text", (40, 70 + row * 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    if angle:
        page = rotate_page(page, angle)
    buffer = io.BytesIO()
    Image.fromarray(page[:, :, ::-1]).save(buffer, format="PNG")
    return buffer.getvalue()


def make_client(tmp_path, **extra):
    settings = Settings(results_dir=tmp_path / "results", samples_dir=tmp_path / "samples",
                        checkpoints_dir=tmp_path / "ckpt", web_dist_dir=tmp_path / "dist")
    paddle = FakePaddle()
    providers = {
        "paddle": lambda: paddle,
        "crnn": lambda: DetectRecognizeEngine(lambda: paddle, FakeLineRecognizer(), "crnn"),
        **extra,
    }
    return TestClient(create_app(settings=settings, engine_providers=providers)), settings


def test_engine_selection_uses_custom_recognizer(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.post("/api/v1/ocr", files={"file": ("a.png", png_bytes(), "image/png")},
                           data={"engine": "crnn"})
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "crnn"
    assert body["pages"][0]["regions"][0]["text"] == "custom model text"
    assert set(body["timings"]) >= {"decode", "preprocess", "recognition", "layout", "postprocess", "total"}
    assert body["stats"]["regions"] == 1


def test_unknown_engine_is_rejected(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.post("/api/v1/ocr", files={"file": ("a.png", png_bytes(), "image/png")},
                           data={"engine": "nope"})
    assert response.status_code == 422


def test_spell_correction_is_reported(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.post("/api/v1/ocr", files={"file": ("a.png", png_bytes(), "image/png")},
                           data={"engine": "paddle", "correct": "true"})
    block = response.json()["pages"][0]["blocks"][0]
    assert block["text"] == "The invoice"
    assert block["original_text"] == "Teh invoice"
    assert block["corrections"][0]["corrected"] == "The"
    assert response.json()["stats"]["corrections"] == 1


def test_preprocessing_stages_are_returned(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.post("/api/v1/ocr", files={"file": ("a.png", png_bytes(4.0), "image/png")},
                           data={"preprocess": "deskew,denoise,contrast"})
    assert response.status_code == 200
    report = response.json()["pages"][0]["preprocessing"]
    assert report["applied"] == ["deskew", "denoise", "contrast"]
    assert [stage["id"] for stage in report["stages"]] == ["original", "deskew", "denoise", "contrast"]
    assert abs(abs(report["deskew_angle"]) - 4.0) < 1.0


def test_invalid_preprocessing_step(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.post("/api/v1/ocr", files={"file": ("a.png", png_bytes(), "image/png")},
                           data={"preprocess": "deskew,magic"})
    assert response.status_code == 422


def test_engines_metrics_and_samples_endpoints(tmp_path):
    client, settings = make_client(tmp_path)
    engines = client.get("/api/v1/engines").json()
    assert {engine["id"] for engine in engines["engines"]} == {"paddle", "crnn"}
    assert engines["preprocessing_steps"] == ["deskew", "denoise", "contrast", "binarize"]

    (settings.results_dir / "evaluation").mkdir(parents=True)
    (settings.results_dir / "evaluation" / "summary.json").write_text(
        json.dumps({"models": {"crnn": {"iam": {"cer": 12.5}}}}), encoding="utf-8")
    metrics = client.get("/api/v1/metrics").json()
    assert metrics["evaluation"]["models"]["crnn"]["iam"]["cer"] == 12.5
    crnn = next(e for e in client.get("/api/v1/engines").json()["engines"] if e["id"] == "crnn")
    assert crnn["metrics"]["iam"]["cer"] == 12.5

    settings.samples_dir.mkdir()
    (settings.samples_dir / "note.png").write_bytes(png_bytes())
    (settings.samples_dir / "manifest.json").write_text(
        json.dumps([{"id": "note", "file": "note.png", "title": "Note"}]), encoding="utf-8")
    assert client.get("/api/v1/samples").json()[0]["id"] == "note"
    assert client.get("/api/v1/samples/note").status_code == 200
    assert client.get("/api/v1/samples/missing").status_code == 404


def test_frontend_is_served_with_spa_fallback(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>DeepOCR UI</html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    client, _ = make_client(tmp_path)
    assert "DeepOCR UI" in client.get("/").text
    assert "DeepOCR UI" in client.get("/models").text
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/api/v1/nothing").status_code == 404


def test_crop_polygon_rectifies_and_pads():
    image = np.zeros((100, 300, 3), np.uint8)
    crop = crop_polygon(image, [[10, 20], [210, 20], [210, 60], [10, 60]])
    assert crop.shape[0] > 40 and crop.shape[1] > 200


def test_skew_estimation_recovers_rotation():
    page = np.full((500, 800, 3), 255, np.uint8)
    for row in range(6):
        cv2.putText(page, "Skewed document text line", (30, 60 + row * 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    rotated = rotate_page(page, -6.0)
    assert abs(estimate_skew(rotated) - 6.0) < 1.0
    processed, report = preprocess_page(rotated, ["deskew"], with_previews=False)
    assert report["applied"] == ["deskew"] and processed.shape[0] > 0


def test_corrector_preserves_numbers_and_acronyms():
    corrector = TextCorrector()
    text, changes = corrector.correct_with_report("Total $1,250.00 for NASA on 2O24")
    assert text == "Total $1,250.00 for NASA on 2024"
    assert [c.kind for c in changes] == ["confusion"]


def test_torch_is_loaded_before_paddle():
    # Regression: on Windows, importing torch after PaddlePaddle fails (WinError 127).
    import sys

    import api  # noqa: F401

    assert "torch" in sys.modules


def test_skew_estimation_survives_shading_and_noise():
    # Regression: a dark lighting gradient used to be thresholded as ink (estimate -13°).
    page = np.full((700, 1000, 3), 240, np.uint8)
    for row in range(8):
        cv2.putText(page, "Notice of the annual meeting text", (40, 70 + row * 75), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 2)
    tilted = rotate_page(page, 4.0).astype(np.float32)
    h, w = tilted.shape[:2]
    tilted *= np.linspace(1.0, 0.55, w)[None, :, None]
    tilted += np.random.default_rng(1).normal(0, 14, tilted.shape)
    degraded = np.clip(tilted, 0, 255).astype(np.uint8)
    assert abs(estimate_skew(degraded) + 4.0) < 1.0


def test_corrector_leaves_names_alone():
    corrector = TextCorrector()
    text, _ = corrector.correct_with_report("Name: Ananya Iyer wrote to Tolch about the paymnet")
    assert text == "Name: Ananya Iyer wrote to Tolch about the payment"


def test_consensus_engine_votes_across_models(tmp_path):
    from api.engines import EngineRegistry, EnsembleEngine

    class Paddle(FakePaddle):
        def recognize_crops(self, crops):
            return [("the quick brown fox", 0.9) for _ in crops]

    class Reader:
        def __init__(self, text, conf):
            self.text, self.conf = text, conf

        def recognize(self, crops):
            return [(self.text, self.conf) for _ in crops]

    paddle = Paddle()
    settings = Settings(results_dir=tmp_path / "results", checkpoints_dir=tmp_path / "ckpt",
                        samples_dir=tmp_path / "samples", web_dist_dir=tmp_path / "dist")
    registry = EngineRegistry(settings, {
        "paddle": lambda: paddle,
        "trocr": lambda: DetectRecognizeEngine(lambda: paddle, Reader("the quick brown fax", 0.95), "trocr"),
        "cnn_bilstm_like": lambda: DetectRecognizeEngine(lambda: paddle, Reader("the quick brown fox", 0.8), "x"),
    })
    engine = EnsembleEngine(registry, {"trocr": 3, "paddle": 2, "cnn_bilstm_like": 2})
    regions = engine.recognize(np.full((100, 300, 3), 255, np.uint8))
    assert regions[0].text == "the quick brown fox"
    last = regions[0].words[-1]
    assert last["text"] == "fox" and set(last["sources"]) == {"paddle", "cnn_bilstm_like"}
    assert "fax" in last["alternatives"]


def test_consensus_needs_two_models(tmp_path):
    from api.engines import EngineRegistry, EnsembleEngine
    from api.ocr_engine import OcrEngineError
    import pytest

    settings = Settings(results_dir=tmp_path / "r", checkpoints_dir=tmp_path / "c")
    registry = EngineRegistry(settings, {"paddle": FakePaddle})
    with pytest.raises(OcrEngineError):
        EnsembleEngine(registry, {"paddle": 1}).recognize(np.zeros((10, 10, 3), np.uint8))


def _wait(client, job_id, timeout=10.0):
    import time as _time
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        body = client.get(f"/api/v1/jobs/{job_id}").json()
        if body["status"] in ("done", "error", "cancelled"):
            return body
        _time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_job_api_reports_progress_and_returns_result(tmp_path):
    client, _ = make_client(tmp_path)
    started = client.post("/api/v1/jobs", files={"file": ("a.png", png_bytes(), "image/png")}, data={"engine": "crnn"})
    assert started.status_code == 202
    body = _wait(client, started.json()["id"])
    assert body["status"] == "done"
    assert body["result"]["engine"] == "crnn"
    assert body["pages"] == 1 and body["lines_total"] == 1 and body["lines_done"] == 1
    assert {"decode", "detect", "recognize", "layout"} <= set(body["stage_times"])


def test_job_api_reports_document_errors(tmp_path):
    client, _ = make_client(tmp_path)
    started = client.post("/api/v1/jobs", files={"file": ("a.txt", b"not an image", "text/plain")})
    body = _wait(client, started.json()["id"])
    assert body["status"] == "error" and body["error_status"] == 415


def test_job_can_be_cancelled_between_line_batches(tmp_path):
    import threading
    gate = threading.Event()

    class SlowReader:
        def recognize(self, crops):
            gate.wait(5)
            return [("slow text", 0.9) for _ in crops]

    class ManyLines(FakePaddle):
        def detect(self, image):
            return [[[10.0, 10 + 30 * i], [200.0, 10 + 30 * i], [200.0, 30 + 30 * i], [10.0, 30 + 30 * i]] for i in range(9)]

    paddle = ManyLines()
    client, _ = make_client(tmp_path, slow=lambda: DetectRecognizeEngine(lambda: paddle, SlowReader(), "slow"))
    job_id = client.post("/api/v1/jobs", files={"file": ("a.png", png_bytes(), "image/png")}, data={"engine": "slow"}).json()["id"]
    assert client.delete(f"/api/v1/jobs/{job_id}").status_code == 200
    gate.set()
    assert _wait(client, job_id)["status"] == "cancelled"
    # The worker released the lock: a new job can start.
    assert client.post("/api/v1/jobs", files={"file": ("a.png", png_bytes(), "image/png")}).status_code == 202
