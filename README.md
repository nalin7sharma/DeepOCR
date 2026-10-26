# DeepOCR — Deep Learning-Based Intelligent OCR for Handwritten and Complex Documents

DeepOCR is an end-to-end OCR system that turns handwritten and printed documents into
structured, editable text. It combines image preprocessing, text detection, layout
analysis, deep-learning text recognition and post-processing in one pipeline, and it
ships with a web studio for reviewing the output.

The project trains and compares three recognition approaches, as planned in the proposal:

| Role | Model | How it was obtained |
| --- | --- | --- |
| Baseline | **CNN + BiLSTM + CTC** | Implemented and trained from scratch in this repository |
| Model 2 | **CRNN** (Shi et al., 2016) | Implemented and trained from scratch in this repository |
| Advanced | **Transformer OCR** (TrOCR: ViT encoder + Transformer decoder) | Pretrained by Microsoft; evaluated zero-shot and fine-tuned on our data (transfer learning) |
| Reference | PaddleOCR (PP-OCR) | Industrial pretrained engine, used as the detector and as a comparison point |

<!-- RESULTS:START -->
## Results

Measured with `scripts/evaluate.py` (2026-10-26T15:16:04) on **IAM handwriting**: 600 held-out test lines; **Synthetic printed + script**: 600 held-out test lines.
CER, WER and accuracy are percentages averaged per line (lower CER/WER is better). ms/line is CPU time.

| Model | Architecture | IAM handwriting CER | WER | Acc | Synthetic printed + script CER | WER | Acc | ms/line |
|---|---|---|---|---|---|---|---|---|
| CNN + BiLSTM (baseline) | CNN-BiLSTM-CTC | 21.79 | 61.97 | 1.00 | 4.85 | 17.49 | 66.67 | 12.2 |
| CNN + BiLSTM (baseline) + post-processing | CNN-BiLSTM-CTC | 20.61 | 51.37 | 4.83 | 4.60 | 15.37 | 69.83 | 42.5 |
| CRNN | CRNN (VGG + BiLSTM + CTC) | 21.66 | 61.47 | 0.50 | 5.25 | 18.54 | 63.17 | 15.9 |
| CRNN + post-processing | CRNN (VGG + BiLSTM + CTC) | 20.55 | 50.99 | 3.17 | 5.00 | 16.13 | 67.00 | 40.5 |
| TrOCR Transformer (zero-shot) | ViT encoder + Transformer decoder | 6.97 | 16.40 | 40.67 | 20.37 | 54.68 | 53.83 | 1829.2 |
| PaddleOCR | PP-OCR recogniser | 28.84 | 72.50 | 1.50 | 3.89 | 21.09 | 66.17 | 478.1 |

- CER/WER/accuracy are averaged per line on the held-out test partitions; lower CER/WER is better.
- Transcriptions are normalised (IAM's tokenised spacing before punctuation is removed) for every model alike; cer_raw keeps the original spacing.
- CNN-BiLSTM and CRNN were trained from scratch on CPU (IAM train + synthetic train only); TrOCR and PaddleOCR are pretrained by their authors.
- TrOCR-small-handwritten was fine-tuned by Microsoft on IAM, so its IAM score may be optimistic if its training partition overlaps this test split.

Charts: `results/plots/model_comparison.png`, `results/plots/robustness.png` and
`results/plots/*_training_curves.png`. All of them are interactive on the web app's **Research** page.
<!-- RESULTS:END -->

## Highlights

- **Real handwriting data**: the IAM Handwriting Database (lines) with its official
  writer-independent train/validation/test split (6,482 / 976 / 2,915 lines).
- **Printed and degraded text**: a synthetic set rendered with 19 printed and 13
  handwriting-style fonts, degraded with noise, blur, JPEG artefacts, skew, uneven
  lighting, ink bleed and inverted (dark-mode) rendering.
- **Consistent preprocessing**: line polarity, brightness and size normalisation is
  shared by training, evaluation and inference (`datasets/line_image.py`), so a model
  sees the same kind of input in the web app as during training.
- **Complete pipeline in the web app**: deskew, denoise, CLAHE and binarise; DBNet
  text-line detection; layout analysis (reading order, titles and headings, tables,
  checkboxes, signature fields); recognition with any of four engines; context-aware
  error correction. Every stage is timed and visualised.
- **Evaluation**: CER, WER, line accuracy and inference time on held-out test sets,
  training curves, sample predictions, and a **robustness study** of CER under noise,
  blur, rotation, low contrast and JPEG compression.
- **Robust engineering**: bounded uploads (size, pages, pixels), one document processed
  at a time with automatic client retry, checkpoints that hot-reload when retrained, an
  OpenCV fallback detector, and 100+ automated tests.

## Quick start (Windows)

Double-click **`start.bat`**, or from PowerShell in the project folder run:

```powershell
.\start.ps1
```

The launcher installs missing Python packages from `web-requirements.txt`, builds the
web UI once (Node.js 20+ is needed for that first build), starts the server, and opens
<http://127.0.0.1:8000>. The API documentation is at <http://127.0.0.1:8000/api/docs>.

### Manual start

```powershell
python -m pip install -r web-requirements.txt
cd web; npm ci; npm run build; cd ..
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

For frontend development with hot reload, run `npm run dev` in `web/` (port 5173). It
proxies `/api` to port 8000.

### Docker

```powershell
Copy-Item .env.example .env
docker compose up --build -d
```

Then open <http://localhost:8080>. See [docs/deployment.md](docs/deployment.md).

### Streamlit research interface

The proposal specifies a Streamlit app. It is in `app/streamlit_app.py` and uses the
same trained checkpoints:

```powershell
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

## Using the web studio

| Page | What it does |
| --- | --- |
| **Studio** | Upload or paste an image or PDF, or pick a one-click demo. Choose an engine and the preprocessing and correction options, then watch the pipeline run. Review the output with structure and confidence-heatmap overlays, a before/after preprocessing slider, a reader, an editor, a line list showing each auto-fix, and a review queue. Export Markdown, text or JSON. |
| **Compare** | Run several engines on the same document. Paste the true text to score each engine (CER/WER) with character-level diffs. |
| **Research** | The measured model comparison, live training curves, the fine-tuning curve, robustness charts, sample predictions, architectures and datasets. |
| **Pipeline** | The proposal's system architecture, stage by stage, with the module that implements each stage. |

Review markers: `[ ]` / `[x]` / `[?]` mark detected checkboxes, `[Signature]` marks a
labelled signature field (it does not verify a signature), and `[Illegible]` marks
low-confidence text. Confidence values are model probabilities, not accuracy.

## System architecture

```text
Document (PDF / image)
   │  api/documents.py        bounded decoding, PDF rendering, EXIF orientation
   ▼
Image preprocessing           api/preprocess.py, preprocessing/image_processor.py
   │  deskew (projection profile), denoise, CLAHE contrast, adaptive binarisation
   ▼
Text detection                api/engines.py (DBNet via PP-OCR), detection/ (OpenCV MSER fallback)
   │  text-line quadrilaterals → perspective-rectified crops
   ▼
Deep-learning recognition     models/ (CNN-BiLSTM, CRNN, TrOCR), models/recognizer.py
   │  line normalisation (datasets/line_image.py) → CTC greedy decode / Transformer decoding
   ▼
Layout analysis               api/structure.py, layout/layout_analyzer.py
   │  reading order, columns, titles and headings, tables, checkboxes, signature fields
   ▼
Post-processing               postprocessing/text_corrector.py
   │  digit/letter confusion fixes, case-preserving spell correction, [Illegible] flags
   ▼
Structured output             Markdown / text / JSON, review UI (web/)
```

## Models

| Model | Architecture | Parameters | Input |
| --- | --- | ---: | --- |
| CNN-BiLSTM | 4 conv blocks (32→64→128→256, BN, ReLU, pooling) → 2-layer BiLSTM (192) → linear → CTC | 2.4 M | 32×512 grey |
| CRNN | 7-layer VGG-style CNN with asymmetric (2×1) pooling → 2× BiLSTM (192) → CTC | 2.8 M | 32×512 grey |
| TrOCR-small | DeiT/ViT encoder (16×16 patches) + Transformer decoder, word-piece tokens | 61 M | 384×384 RGB |

The CTC models were trained on CPU because no GPU was available, which is why their
channel widths are reduced (the CRNN paper uses 64–512). The topologies match the
papers; widths are set in `configs/model.yaml`. A custom ViT encoder-decoder is also
implemented (`models/transformer_ocr.py`, `CustomViTOCR`). The experiments use
pretrained TrOCR instead, because a Transformer trained from scratch needs far more
data and compute than a CPU allows (the proposal's "transfer learning where appropriate").

Training details: Adam (lr 1e-3, weight decay 1e-4), cosine schedule, gradient
clipping 5, batch 32, early stopping on validation CER, and on-the-fly augmentation
(rotation, brightness, contrast, Gaussian noise, blur, scaling without cropping,
perspective).

## Datasets

| Set | Train | Val | Test | Notes |
| --- | ---: | ---: | ---: | --- |
| IAM handwriting (lines) | 6,482 | 976 | 2,915 | Official writer-independent split, [Teklia/IAM-line](https://huggingface.co/datasets/Teklia/IAM-line) export |
| Synthetic printed + script | 5,005 | 520 | 975 | `scripts/prepare_dataset.py`, 32 fonts with document degradations |

Test sets are never used for training or model selection. Transcriptions are
normalised the same way for every model (IAM tokenises punctuation as "word ." and
this is undone). The raw-spacing CER is also reported as `cer_raw`.

## Reproducing the experiments

```powershell
python -m pip install -r requirements.txt            # research dependencies (PyTorch etc.)
python scripts/download_iam_hf.py                     # IAM lines → data/processed/iam
python scripts/prepare_dataset.py --dataset synthetic --n_samples 6500
python scripts/train.py --model cnn_bilstm --epochs 11 --max_minutes 170
python scripts/train.py --model crnn --epochs 9 --max_minutes 170
python scripts/finetune_trocr.py --steps 300          # transfer learning (optional)
python scripts/evaluate.py --all --limit 600          # held-out test sets → results/
python scripts/robustness.py --lines 120              # degradation study → results/
python scripts/make_samples.py                        # demo documents for the web app
```

Training writes `checkpoints/<model>/best_model.pth` (selected on validation CER). The
checkpoint stores the architecture, charset and input size, so inference rebuilds the
exact model. Training logs and curves go to `results/training/` and `results/plots/`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/ocr` | OCR a multipart upload: `file`, optional `engine` (`paddle`, `crnn`, `cnn_bilstm`, `trocr`), `preprocess` (`deskew,denoise,contrast,binarize`), `correct` (`true`/`false`) |
| `GET` | `/api/v1/engines` | Engine availability, model info and evaluation metrics |
| `GET` | `/api/v1/metrics` | Evaluation summary, training histories, robustness and sample predictions |
| `GET` | `/api/v1/samples` | Built-in demo documents |
| `GET` | `/api/v1/health/live`, `/api/v1/health/ready` | Liveness and readiness |
| `GET` | `/api/docs` | Interactive OpenAPI documentation |

The OCR response contains page previews, blocks (kind, text, Markdown, polygon,
confidence, review flag, corrections), raw regions, preprocessing stages, per-stage
timings and document statistics. Errors: `413` (size/page/pixel limits), `415`
(unsupported type), `422` (invalid/encrypted file or option), `429` (busy; the UI
retries automatically), `503` (engine unavailable).

## Tests

```powershell
python -m pytest tests api/tests -q        # 100 backend, model, pipeline and API tests
cd web; npm test; npm run build            # frontend tests and production build
```

## Project structure

```text
DeepOCR/
├── api/              FastAPI service: decoding, preprocessing, engines, layout, endpoints
├── app/              Streamlit research interface
├── checkpoints/      Trained weights (best_model.pth per model, fine-tuned TrOCR)
├── configs/          Dataset, model and training configuration
├── data/             Prepared datasets (generated; see Reproducing the experiments)
├── datasets/         Dataset loaders, augmentation, shared line normalisation, splits
├── detection/        OpenCV MSER text detector (fallback)
├── evaluation/       CER / WER / accuracy metrics
├── inference/        End-to-end research pipeline (CLI / Streamlit)
├── layout/           Layout analysis (roles, columns, reading order)
├── models/           CNN-BiLSTM, CRNN, Transformer OCR, recogniser wrappers
├── postprocessing/   Context-aware error correction
├── preprocessing/    Image preprocessing operations
├── results/          Evaluation JSON/CSV/plots, training histories
├── samples/          Demo documents for the web app
├── scripts/          Data, training, fine-tuning, evaluation, robustness, samples
├── tests/            Research pipeline tests
├── training/         Trainer and CTC utilities
├── web/              React + TypeScript + Vite web studio
├── start.ps1 / .bat  One-click launcher
└── docker-compose.yml
```

## Limitations

- The CTC models were trained for about three hours on a CPU. More epochs, a GPU, wider
  layers, and a language model with beam search would lower their error further.
- TrOCR-small-handwritten was fine-tuned by Microsoft on IAM. Its IAM split may differ
  from the one used here, so its IAM score can be optimistic.
- Layout reconstruction (tables, checkboxes, headings, columns) is heuristic and is
  flagged for human review. It has not been benchmarked against a labelled layout dataset.
- English only. Hindi and other scripts are future work.
- The default deployment has no authentication or TLS. Use it on localhost or behind a
  secured proxy.

## Future scope

Multilingual and Hindi handwriting recognition, mathematical expressions, learned
layout detection (e.g. LayoutLM / DocTR), signature detection, document
classification, mobile and real-time camera OCR, and vision-language model integration.
