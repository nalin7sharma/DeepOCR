# Self-hosted web deployment

DeepOCR's team web application consists of a React/Vite frontend served by unprivileged Nginx and a FastAPI service using pretrained PaddleOCR. The default Compose setup is CPU-only, requires outbound network access on first start to download English OCR weights, and persists those model weights in the `paddle-model-cache` volume. The pretrained English model is intended for printed/scene text; handwriting accuracy varies and should be validated on representative data.

## Start and update

```bash
cp .env.example .env
docker compose up --build -d
docker compose ps
docker compose logs -f api
```

Open `http://localhost:8080`. The API readiness check initializes PaddleOCR, so the first start may take several minutes while the model weights are downloaded. Later starts reuse the cache. The API is reachable only on the internal Compose network; Nginx exposes the UI and proxies `/api/` to FastAPI.

On Windows PowerShell, use `Copy-Item .env.example .env`. The default binding is `127.0.0.1:8080`. For team access, terminate TLS at a trusted reverse proxy and restrict network access using your firewall/VPN. If the proxy is on another host, configure `DEEP_OCR_BIND_ADDRESS` accordingly and do not expose the HTTP port to an untrusted network. This deployment intentionally has no user accounts or built-in TLS.

## Resource limits and uploads

Settings may be placed in `.env`:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `DEEP_OCR_BIND_ADDRESS` | `127.0.0.1` | Host interface used for the frontend port |
| `DEEP_OCR_PORT` | `8080` | Host port |
| `OCR_MAX_UPLOAD_BYTES` | `104857600` | Maximum upload file size (100 MiB) |
| `OCR_MAX_PAGES` | `100` | Maximum PDF pages or TIFF frames |
| `OCR_MAX_IMAGE_DIMENSION` | `10000` | Maximum width or height for uploaded raster images |
| `OCR_MAX_PIXELS_PER_PAGE` | `40000000` | Maximum decoded pixels per page |
| `OCR_MAX_TOTAL_PIXELS` | `100000000` | Maximum combined decoded pixels across the whole document |
| `OCR_PDF_RENDER_SCALE` | `2` | PDF rasterization scale, reduced automatically to meet limits |
| `OCR_REVIEW_CONFIDENCE_THRESHOLD` | `0.45` | PaddleOCR confidence below which a text line is shown as `[Illegible]` and flagged for review |
| `OCR_LANGUAGE` | `en` | PaddleOCR language model |
| `API_MEMORY_LIMIT` | `6g` | Container memory limit; tune for host and workload |
| `API_CPU_LIMIT` | `4.0` | Container CPU limit |
| `WEB_MEMORY_LIMIT` | `128m` | Frontend container memory limit |

The maximum uploaded file is 100 MiB and the maximum PDF/TIFF document is 100 pages by default. Nginx allows a 105 MiB request body to accommodate multipart form overhead, while FastAPI enforces `OCR_MAX_UPLOAD_BYTES` against the file itself. If you raise the API file limit, raise `client_max_body_size` in `web/nginx.conf` accordingly. Raster dimensions, per-page pixels, and aggregate document pixels remain bounded before decoded pages are retained or passed to OCR, so a large or high-resolution document can still be rejected by those safeguards even when below the page-count limit. Unsupported, corrupt, and encrypted PDFs receive explicit errors. A single API process handles one document at a time and responds with `429` while busy, avoiding concurrent model/image memory spikes. Scale by provisioning a suitably sized host and adjusting limits before adding workers.

## Document privacy and operations

- Uploaded content, rendered pages, and OCR output are request-scoped and are not written to a document/result database or included in application logs. Upload streams close after each request; the API container uses a bounded `/tmp` tmpfs for short-lived multipart buffering.
- Only OCR model weights are persisted, in the `paddle-model-cache` Docker volume. Back up or remove that volume independently of documents.
- Docker containers run as non-root, the API filesystem is read-only except for the model cache and temporary filesystem, and the API port is not published on the host.
- Health endpoints: `/api/v1/health/live` and `/api/v1/health/ready`. The readiness check initializes the model and returns an error if the runtime or model download is unavailable.
- API schema and interactive docs: `/api/openapi.json` and `/api/docs`.
- Logs contain operational errors only. Avoid enabling request-body logging in external proxies or observability agents.

`docker compose down` stops the service while retaining model weights. To remove the model cache, explicitly remove the `paddle-model-cache` volume after stopping the stack; no document data is stored there.

## Structured output and review

Each page response contains a bounded JPEG source preview, ordered OCR blocks, original text regions/confidences, and editable-ready Markdown. Tables are reconstructed from ruled grid cells or repeated, widely separated OCR cells; inferred tables preserve blank cells and are always flagged for review. `[ ]` and `[x]` mark visually detected unchecked/checked boxes; `[?]` marks an ambiguous detected box. Every checkbox classification is flagged for human review; its confidence is a heuristic, not a calibrated OCR probability. Printed signature labels produce a `[Signature]` field marker; this does not recognize handwriting, prove a signature is present, or verify a signer. Text below `OCR_REVIEW_CONFIDENCE_THRESHOLD` is rendered as `[Illegible]`, while the original OCR text and confidence remain available to reviewers. Downloads use the edited Markdown currently shown in the browser.

Layout reconstruction is a visual heuristic, not a guarantee of exact form or table geometry. Compare uncertain results against the original page before reusing them.

## Local development and tests

Install web dependencies with `python -m pip install -r web-requirements.txt`, then run `python -m uvicorn api.main:app --reload --port 8000`. In `web/`, run `npm ci` and `npm run dev`; Vite proxies API requests to port 8000. The first OCR request/readiness check downloads model weights to PaddleOCR's local cache.

Run backend tests from the project root with `python -m pytest api/tests tests/ -q`. Run frontend tests and verify the optimized bundle with `cd web && npm test && npm run build`.

## Research models

The CNN+BiLSTM, CRNN, and custom Transformer OCR training/evaluation scripts remain available in the repository and are not loaded by the production API. The Streamlit research/demo application and CLI pipeline require a trained checkpoint at `checkpoints/<model>/best_model.pth`; missing or invalid weights are reported as errors rather than producing random predictions.

## Recognition engines

`GET /api/v1/engines` lists every engine with its availability, model details and the
test-set metrics from `results/evaluation/summary.json`. The client picks one per
request with the `engine` form field:

| Engine | Detection | Recognition | Needs |
| --- | --- | --- | --- |
| `paddle` | DBNet (PP-OCR) | PP-OCR English recogniser | PaddleOCR weights (downloaded once) |
| `cnn_bilstm` | DBNet, or OpenCV MSER if Paddle is unavailable | CNN-BiLSTM + CTC trained in this project | `checkpoints/cnn_bilstm/best_model.pth` |
| `crnn` | DBNet, or OpenCV MSER if Paddle is unavailable | CRNN + CTC trained in this project | `checkpoints/crnn/best_model.pth` |
| `trocr` | DBNet | TrOCR-small (ViT + Transformer), pretrained | Hugging Face download (cached in `hf-model-cache`) |
| `trocr_ft` | DBNet | TrOCR-small fine-tuned on IAM + synthetic | `checkpoints/trocr_finetuned/` |

Checkpoints are reloaded automatically when retraining writes a newer file. A failed
load is reported in the engine's `error` field and retried on the next request.
`preprocess` (`deskew,denoise,contrast,binarize`) and `correct` (`true`/`false`) are
optional per request.

## Single-process mode (no Docker)

When `web/dist` exists, FastAPI serves the built UI itself, so the whole app runs on
one port: `python -m uvicorn api.main:app --port 8000`, or just `start.ps1`. Bind to
`127.0.0.1` unless the host is protected by a firewall or VPN.
