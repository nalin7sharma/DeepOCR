"""FastAPI application for the transient DeepOCR web service."""
from __future__ import annotations

import json
import logging
import base64
import io
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

from api import progress
from api.documents import DocumentError, decode_document
from api.engines import CATALOG, EngineRegistry
from api.ocr_engine import OcrEngine, OcrEngineError, TextRegion
from api.preprocess import STEP_ORDER, preprocess_page
from api.settings import Settings, get_settings
from api.structure import reconstruct_page

logger = logging.getLogger("deepocr.api")

CORRECTABLE_KINDS = {"text", "heading", "title"}
MAX_STAGE_PREVIEW_PAGES = 3


class WordVote(BaseModel):
    text: str
    support: float | None = None
    sources: list[str] = Field(default_factory=list)
    alternatives: dict[str, float] = Field(default_factory=dict)


class RegionResponse(BaseModel):
    polygon: list[list[float]]
    text: str
    confidence: float = Field(ge=0, le=1)
    review_required: bool = False
    words: list[WordVote] | None = None


class CorrectionResponse(BaseModel):
    original: str
    corrected: str
    kind: str
    position: int


class BlockResponse(BaseModel):
    kind: str
    text: str
    markdown: str
    polygon: list[list[float]]
    confidence: float = Field(ge=0, le=1)
    review_required: bool
    raw_text: str | None = None
    cells: list[list[str]] | None = None
    checkbox_state: str | None = None
    original_text: str | None = None
    corrections: list[CorrectionResponse] = Field(default_factory=list)
    words: list[WordVote] | None = None


class StageResponse(BaseModel):
    id: str
    label: str
    ms: float
    image: str


class PreprocessingResponse(BaseModel):
    applied: list[str] = Field(default_factory=list)
    deskew_angle: float | None = None
    stages: list[StageResponse] = Field(default_factory=list)


class PageResponse(BaseModel):
    page_number: int
    text: str
    markdown: str
    width: int
    height: int
    preview_image: str
    blocks: list[BlockResponse]
    regions: list[RegionResponse]
    preprocessing: PreprocessingResponse = Field(default_factory=PreprocessingResponse)
    timings: dict[str, float] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    pages: int
    lines: int
    words: int
    characters: int
    regions: int
    mean_confidence: float
    low_confidence: int
    review_items: int
    corrections: int
    confidence_histogram: list[int]


class OcrResponse(BaseModel):
    filename: str
    model: str
    engine: str = "paddle"
    language: str
    inference_ms: float
    text: str
    markdown: str
    pages: list[PageResponse]
    timings: dict[str, float] = Field(default_factory=dict)
    stats: StatsResponse | None = None
    options: dict = Field(default_factory=dict)


def _preview_image(image) -> str:
    preview = Image.fromarray(image[:, :, ::-1])
    preview.thumbnail((1000, 1400), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    preview.save(output, format="JPEG", quality=72, optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _safe_filename(filename: str | None) -> str:
    name = (filename or "document").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "_", name).strip(" .")
    return name[:255] or "document"


def _parse_steps(value: str | None) -> list[str]:
    if not value or value.strip().lower() in {"", "none", "off", "false"}:
        return []
    if value.strip().lower() in {"auto", "default", "on", "true"}:
        return ["deskew", "contrast"]
    steps = [step.strip().lower() for step in value.split(",") if step.strip()]
    unknown = [step for step in steps if step not in STEP_ORDER]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown preprocessing step(s): {', '.join(unknown)}")
    return steps


def _apply_corrections(blocks: list[dict], corrector) -> int:
    count = 0
    for block in blocks:
        if block["kind"] not in CORRECTABLE_KINDS or block.get("review_required"):
            continue
        corrected, changes = corrector.correct_with_report(block["text"])
        if not changes:
            continue
        prefix = block["markdown"][: len(block["markdown"]) - len(block["text"])] if block["markdown"].endswith(block["text"]) else ""
        block["original_text"] = block["text"]
        block["text"] = corrected
        block["markdown"] = f"{prefix}{corrected}"
        block["corrections"] = [change.to_dict() for change in changes]
        count += len(changes)
    return count


def _document_stats(pages: list[PageResponse], corrections: int, threshold: float) -> StatsResponse:
    confidences = [region.confidence for page in pages for region in page.regions]
    text_blocks = [b for page in pages for b in page.blocks if b.kind != "table"]
    words = sum(len(b.text.split()) for page in pages for b in page.blocks)
    characters = sum(len(b.text) for page in pages for b in page.blocks)
    histogram = np.histogram(confidences, bins=10, range=(0, 1))[0].tolist() if confidences else [0] * 10
    return StatsResponse(
        pages=len(pages),
        lines=len(text_blocks),
        words=words,
        characters=characters,
        regions=len(confidences),
        mean_confidence=round(float(np.mean(confidences)), 4) if confidences else 0.0,
        low_confidence=sum(1 for c in confidences if c < threshold),
        review_items=sum(1 for page in pages for b in page.blocks if b.review_required),
        corrections=corrections,
        confidence_histogram=histogram,
    )


def _process_document(
    filename: str,
    content: bytes,
    settings: Settings,
    engine: OcrEngine,
    engine_id: str,
    steps: list[str],
    correct: bool,
) -> OcrResponse:
    started = time.perf_counter()
    progress.report("decode")
    pages = decode_document(content, settings)
    progress.report(pages=len(pages))
    decode_ms = (time.perf_counter() - started) * 1000
    corrector = None
    if correct:
        from postprocessing.text_corrector import TextCorrector

        corrector = TextCorrector()

    totals = {"decode": decode_ms, "preprocess": 0.0, "recognition": 0.0, "layout": 0.0, "postprocess": 0.0}
    page_results = []
    correction_count = 0
    for page in pages:
        timings: dict[str, float] = {}
        mark = time.perf_counter()
        progress.report("preprocess", page=page.page_number)
        image, report = preprocess_page(
            page.image, steps, with_previews=page.page_number <= MAX_STAGE_PREVIEW_PAGES
        )
        timings["preprocess"] = (time.perf_counter() - mark) * 1000

        mark = time.perf_counter()
        regions: list[TextRegion] = engine.recognize(image)
        timings["recognition"] = (time.perf_counter() - mark) * 1000

        mark = time.perf_counter()
        progress.report("layout")
        extraction = reconstruct_page(image, regions, review_threshold=settings.review_confidence_threshold)
        blocks = [block.to_dict() for block in extraction.blocks]
        timings["layout"] = (time.perf_counter() - mark) * 1000

        mark = time.perf_counter()
        progress.report("postprocess")
        markdown = extraction.markdown
        if corrector is not None:
            correction_count += _apply_corrections(blocks, corrector)
            markdown = "\n".join(block["markdown"] for block in blocks)
        timings["postprocess"] = (time.perf_counter() - mark) * 1000

        for key, value in timings.items():
            totals[key] += value
        page_results.append(
            PageResponse(
                page_number=page.page_number,
                text=markdown,
                markdown=markdown,
                width=image.shape[1],
                height=image.shape[0],
                preview_image=_preview_image(image),
                blocks=[BlockResponse(**block) for block in blocks],
                regions=[
                    RegionResponse(
                        polygon=region.polygon,
                        text=region.text,
                        confidence=min(1.0, max(0.0, region.confidence)),
                        review_required=region.confidence < settings.review_confidence_threshold,
                        words=region.words,
                    )
                    for region in regions
                ],
                preprocessing=PreprocessingResponse(**report),
                timings={key: round(value, 1) for key, value in timings.items()},
            )
        )
    document_markdown = "\n\n".join(
        f"## Page {page.page_number}\n\n{page.markdown}".rstrip() for page in page_results
    )
    total_ms = round((time.perf_counter() - started) * 1000, 1)
    totals["total"] = total_ms
    spec = CATALOG.get(engine_id)
    return OcrResponse(
        filename=filename,
        model="PaddleOCR" if engine_id == "paddle" else (spec.label if spec else engine_id),
        engine=engine_id,
        language=settings.language,
        inference_ms=total_ms,
        text=document_markdown,
        markdown=document_markdown,
        pages=page_results,
        timings={key: round(value, 1) for key, value in totals.items()},
        stats=_document_stats(page_results, correction_count, settings.review_confidence_threshold),
        options={"engine": engine_id, "preprocess": steps, "correct": correct},
    )


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def create_app(
    settings: Settings | None = None,
    engine_provider: Callable[[], OcrEngine] | None = None,
    engine_providers: dict[str, Callable[[], OcrEngine]] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    overrides = engine_providers
    if engine_provider is not None:
        overrides = {"paddle": engine_provider, **(engine_providers or {})}
    registry = EngineRegistry(settings, overrides)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.warmup:
            threading.Thread(
                target=registry.warmup, args=([settings.default_engine, "crnn", "cnn_bilstm"],),
                daemon=True, name="deepocr-warmup",
            ).start()
        yield

    app = FastAPI(
        title="DeepOCR API",
        version="2.0.0",
        description="Transient OCR for images and multi-page PDFs with pretrained and custom deep-learning engines.",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.registry = registry
    app.state.processing_lock = threading.Lock()

    @app.get("/api/v1/health/live", tags=["health"])
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/health/ready", tags=["health"])
    def ready() -> dict[str, str]:
        try:
            registry.get(settings.default_engine if settings.default_engine in registry.ids() else registry.ids()[0])
        except OcrEngineError:
            logger.exception("OCR engine readiness check failed.")
            raise HTTPException(status_code=503, detail="OCR engine is unavailable.") from None
        return {"status": "ready"}

    @app.get("/api/v1/engines", tags=["ocr"])
    def engines() -> dict:
        return {
            "default": settings.default_engine,
            "busy": app.state.processing_lock.locked(),
            "engines": registry.status(),
            "preprocessing_steps": list(STEP_ORDER),
        }

    @app.get("/api/v1/metrics", tags=["research"])
    def metrics() -> dict:
        results = settings.results_dir
        training = {}
        for path in sorted((results / "training").glob("*_history.json")):
            payload = _read_json(path)
            if payload:
                training[payload.get("model", path.stem)] = payload
        return {
            "evaluation": _read_json(results / "evaluation" / "summary.json"),
            "samples": _read_json(results / "evaluation" / "samples.json"),
            "robustness": _read_json(results / "evaluation" / "robustness.json"),
            "ensemble": _read_json(results / "evaluation" / "ensemble.json"),
            "training": training,
        }

    def _sample_manifest() -> list[dict]:
        payload = _read_json(settings.samples_dir / "manifest.json")
        return payload if isinstance(payload, list) else []

    @app.get("/api/v1/samples", tags=["samples"])
    def samples() -> list[dict]:
        return _sample_manifest()

    def _sample_path(sample_id: str, key: str) -> Path:
        for sample in _sample_manifest():
            if sample.get("id") == sample_id and sample.get(key):
                path = (settings.samples_dir / sample[key]).resolve()
                if settings.samples_dir.resolve() in path.parents and path.is_file():
                    return path
        raise HTTPException(status_code=404, detail="Sample not found.")

    @app.get("/api/v1/samples/{sample_id}", tags=["samples"])
    def sample_file(sample_id: str):
        path = _sample_path(sample_id, "file")
        return FileResponse(path, filename=path.name)

    @app.get("/api/v1/samples/{sample_id}/thumb", tags=["samples"], include_in_schema=False)
    def sample_thumb(sample_id: str):
        try:
            return FileResponse(_sample_path(sample_id, "thumb"))
        except HTTPException:
            return FileResponse(_sample_path(sample_id, "file"))

    @app.post("/api/v1/ocr", response_model=OcrResponse, tags=["ocr"])
    async def recognize(
        file: UploadFile = File(...),
        engine: str | None = Form(None),
        preprocess: str | None = Form(None),
        correct: bool = Form(False),
    ) -> OcrResponse:
        engine_id = (engine or settings.default_engine).strip().lower()
        if engine_id not in registry.ids():
            await file.close()
            raise HTTPException(
                status_code=422,
                detail=f"Unknown engine '{engine_id}'. Available: {', '.join(registry.ids())}.",
            )
        steps = _parse_steps(preprocess)
        try:
            content = bytearray()
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                content.extend(chunk)
                if len(content) > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"The upload exceeds the {settings.max_upload_bytes}-byte limit.",
                    )
            if not content:
                raise HTTPException(status_code=400, detail="The uploaded file is empty.")
            if not app.state.processing_lock.acquire(blocking=False):
                raise HTTPException(
                    status_code=429,
                    detail="Another document is being processed. Retry shortly.",
                    headers={"Retry-After": "3"},
                )
            try:
                try:
                    selected = await run_in_threadpool(registry.get, engine_id)
                    return await run_in_threadpool(
                        _process_document,
                        _safe_filename(file.filename),
                        bytes(content),
                        settings,
                        selected,
                        engine_id,
                        steps,
                        correct,
                    )
                except DocumentError as exc:
                    raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
                except OcrEngineError as exc:
                    logger.exception("OCR processing failed.")
                    detail = (
                        "OCR processing failed. Check the server's model cache and logs."
                        if engine_id == "paddle"
                        else f"The {engine_id} engine is unavailable: {exc}"
                    )
                    raise HTTPException(status_code=503, detail=detail) from exc
            finally:
                app.state.processing_lock.release()
        finally:
            await file.close()

    # ---------------------------------------------------------------- jobs
    store = progress.JobStore()
    app.state.jobs = store

    def _engine_error_detail(engine_id: str, exc: Exception) -> str:
        if engine_id == "paddle":
            return "OCR processing failed. Check the server's model cache and logs."
        return f"The {engine_id} engine is unavailable: {exc}"

    @app.post("/api/v1/jobs", status_code=202, tags=["ocr"])
    async def create_job(
        file: UploadFile = File(...),
        engine: str | None = Form(None),
        preprocess: str | None = Form(None),
        correct: bool = Form(False),
    ) -> dict:
        """Start OCR in the background; poll ``GET /api/v1/jobs/{id}`` for live progress and the result."""
        engine_id = (engine or settings.default_engine).strip().lower()
        try:
            if engine_id not in registry.ids():
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown engine '{engine_id}'. Available: {', '.join(registry.ids())}.",
                )
            steps = _parse_steps(preprocess)
            content = bytearray()
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                content.extend(chunk)
                if len(content) > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"The upload exceeds the {settings.max_upload_bytes}-byte limit.",
                    )
            if not content:
                raise HTTPException(status_code=400, detail="The uploaded file is empty.")
            filename = _safe_filename(file.filename)
        finally:
            await file.close()
        if not app.state.processing_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="Another document is being processed. Retry shortly.",
                headers={"Retry-After": "3"},
            )
        job = store.add(progress.Job(filename=filename, engine=engine_id))
        payload = bytes(content)

        def run() -> None:
            progress.bind(job)
            try:
                selected = registry.get(engine_id)
                result = _process_document(filename, payload, settings, selected, engine_id, steps, correct)
                job.finish("done", result=result.model_dump())
            except progress.JobCancelled:
                job.finish("cancelled")
            except DocumentError as exc:
                job.finish("error", error=str(exc), error_status=exc.status_code)
            except OcrEngineError as exc:
                logger.exception("OCR job failed.")
                job.finish("error", error=_engine_error_detail(engine_id, exc), error_status=503)
            except Exception:  # pragma: no cover - unexpected failure must not leak details
                logger.exception("Unexpected OCR job failure.")
                job.finish("error", error="Unexpected processing error. See the server log.", error_status=500)
            finally:
                progress.bind(None)
                app.state.processing_lock.release()

        threading.Thread(target=run, daemon=True, name=f"ocr-job-{job.id[:8]}").start()
        return job.snapshot(include_result=False)

    @app.get("/api/v1/jobs/{job_id}", tags=["ocr"])
    def job_status(job_id: str) -> dict:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found or expired.")
        return job.snapshot()

    @app.delete("/api/v1/jobs/{job_id}", tags=["ocr"])
    def cancel_job(job_id: str) -> dict:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found or expired.")
        with job.lock:
            if job.status in ("queued", "running"):
                job.cancelled = True
        return job.snapshot(include_result=False)

    _mount_frontend(app, settings.web_dist_dir)
    return app


def _mount_frontend(app: FastAPI, dist: Path) -> None:
    """Serve the built React UI from the API process (single-port deployment)."""
    index = dist / "index.html"
    if not index.is_file():
        return
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        candidate = (dist / full_path).resolve()
        if full_path and dist.resolve() in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)


app = create_app()
