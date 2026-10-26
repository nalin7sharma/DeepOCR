# DeepOCR implementation notes

What exists, how it was verified, and where to find it.

## Implemented

- **Data**: IAM lines (official split, `scripts/download_iam_hf.py`) and a degraded
  synthetic printed/script set (`scripts/prepare_dataset.py`). One line-normalisation
  function is shared by training and inference (`datasets/line_image.py`).
- **Models**: CNN-BiLSTM and CRNN (CTC), trained here on CPU (`scripts/train.py`).
  TrOCR (ViT + Transformer) is evaluated zero-shot and fine-tuned (`scripts/finetune_trocr.py`).
  Checkpoints carry their own architecture metadata (`models/recognizer.py`).
- **Evaluation**: CER, WER, accuracy and speed on held-out test sets (`scripts/evaluate.py`),
  plus a robustness study under synthetic degradations (`scripts/robustness.py`).
- **Web app**: FastAPI with four engines, preprocessing stages, layout reconstruction
  and error correction, plus a React studio (Studio, Compare, Research, Pipeline). It
  runs as a single process (`start.ps1`) or under Docker Compose.
- **Streamlit**: `app/streamlit_app.py` runs the research pipeline with the trained checkpoints.

## Fixes made during hardening

- The inference pipeline passed un-resized crops to height-specific models; it now uses
  the shared normalisation and batch recognition.
- The scale augmentation cropped text while keeping the full label, so images and labels
  disagreed; it now pads or stretches instead.
- The spell corrector turned every `0` into `O` and every `|` into `l`, and lowercased words.
  It now makes context-aware, case-preserving fixes and reports each one.
- The layout analyzer split ragged single-column text into fake columns; it now detects
  real gutters.
- The Transformer training path was inconsistent (wrong special tokens, wrong output types);
  the Transformer experiments now use TrOCR fine-tuning.
- Engine checkpoints hot-reload after retraining, and a failed load is retried instead
  of disabling the engine.
- On Windows, loading PaddlePaddle before PyTorch made `import torch` fail (WinError 127),
  which silently disabled the trained engines in the server. The API now loads torch first.
- TrOCR fine-tuning used the wrong decoder start token and label layout, so its loss sat at
  random-guess level (~11.7). It now uses token 2 as the start and drops the leading `<s>`
  from labels (initial loss 0.83).
- PP-OCR's default detection threshold (0.6) missed faint handwritten lines; it is now
  0.5 and configurable (`OCR_DET_BOX_THRESH`).
- Deskew misread uneven lighting as ink (−13° instead of −4.2° on the noisy demo). Skew
  estimation now flat-fields the page first.
- Line grouping used axis-aligned boxes, so tilted lines merged into one. It now follows
  each region's rotated baseline.

## Verification

`python -m pytest tests api/tests -q` (backend), `cd web && npm test && npm run build`
(frontend), plus end-to-end runs of every demo document through the web app.
