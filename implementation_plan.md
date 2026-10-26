# Deep Learning OCR System — Implementation Plan

## Overview
This document records the original plan for the modular deep learning OCR research pipeline. The repository now contains the training, evaluation, preprocessing, layout, and custom-model components, plus a self-hosted web application built with FastAPI, React, and Docker Compose. The pretrained PaddleOCR backend is the web default; custom model inference requires trained checkpoints.

### Structured OCR review
- Preserve page previews, ordered text regions, confidence, and layout in the API response.
- Emit conservative Markdown for ruled/repeated-cell tables, checkboxes, signature labels, and low-confidence text; flag heuristic or uncertain structure for human review.
- Provide an editable source-aligned review UI with page navigation, region selection, and corrected Markdown/text/JSON export.
- Signature placeholders identify labeled fields only and never assert signer or signature verification.
- Measure the production OCR backend only against labeled, held-out samples; do not treat model confidence as recognition accuracy.
- Use one deterministic train/validation/test assignment for both custom-model training and evaluation, and never report evaluation metrics from untrained weights.

## Proposed Project Structure

```
DeepOCR/
├── api/                        # FastAPI API and PaddleOCR backend
├── web/                        # React + TypeScript frontend, built and served by Nginx
├── app/
│   └── streamlit_app.py          # Research/demo UI
├── configs/
│   ├── default.yaml              # Global config
│   ├── dataset.yaml              # Dataset paths & splits
│   ├── model.yaml                # Model hyperparameters
│   └── training.yaml             # Training settings
├── data/
│   ├── raw/                      # Downloaded datasets
│   ├── processed/                # Preprocessed images
│   └── README.md                 # Dataset setup guide
├── datasets/
│   ├── __init__.py
│   ├── base_dataset.py           # Abstract base class
│   ├── ocr_dataset.py            # General OCR dataset
│   └── augmentations.py          # All augmentation logic
├── models/
│   ├── __init__.py
│   ├── cnn_bilstm.py             # CNN + BiLSTM + CTC
│   ├── crnn.py                   # CRNN (Shi et al.)
│   └── transformer_ocr.py        # ViT Encoder + Decoder
├── preprocessing/
│   ├── __init__.py
│   └── image_processor.py        # All preprocessing ops
├── detection/
│   ├── __init__.py
│   ├── base_detector.py          # Abstract detector
│   └── opencv_detector.py        # OpenCV MSER/contour detector
├── layout/
│   ├── __init__.py
│   └── layout_analyzer.py        # Reading order, blocks
├── postprocessing/
│   ├── __init__.py
│   └── text_corrector.py         # Spell/error correction
├── training/
│   ├── __init__.py
│   ├── trainer.py                # Generic trainer class
│   └── ctc_utils.py              # CTC decoder + beam search
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                # CER, WER, accuracy, time
│   └── evaluator.py              # Full eval pipeline
├── inference/
│   ├── __init__.py
│   └── ocr_pipeline.py           # End-to-end inference
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── checkpoint.py
│   └── visualization.py
├── checkpoints/                  # Saved model weights
├── results/                      # Eval outputs, plots
├── scripts/
│   ├── download_dataset.py       # IAM dataset download helper
│   ├── prepare_dataset.py        # Build processed splits
│   ├── train.py                  # Main training entry
│   ├── evaluate.py               # Main evaluation entry
│   └── compare_models.py         # Model comparison table
├── tests/
│   ├── test_preprocessing.py
│   ├── test_metrics.py
│   ├── test_models.py
│   ├── test_dataset.py
│   └── test_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Implementation Phases

### Phase 1 — Config & Utils
- YAML configs with OmegaConf/PyYAML
- Logger, checkpoint saver, seed utils

### Phase 2 — Preprocessing
- Full OpenCV + PIL preprocessing pipeline
- 8 operations: resize, grayscale, denoise, contrast, binarize, deskew, normalize, unblur

### Phase 3 — Datasets & Augmentation
- Base dataset class with character vocabulary
- IAM lines dataset loader
- 7 augmentation types via torchvision/albumentations

### Phase 4 — Models
- CNN+BiLSTM+CTC (baseline)
- CRNN (Shi et al. architecture)
- Transformer OCR (ViT encoder + autoregressive decoder)

### Phase 5 — Training
- Generic trainer with CTC and cross-entropy loss paths
- LR scheduler, gradient clipping, best-model checkpoint

### Phase 6 — Detection & Layout
- OpenCV MSER + contour-based text detector
- Rule-based layout analyzer (reading order, block classification)

### Phase 7 — Post-Processing
- Character substitution rules
- SymSpell / pyspellchecker word correction
- Confidence-based filtering

### Phase 8 — Evaluation
- CER, WER, accuracy, inference time
- Compare script generating markdown/CSV tables

### Phase 9 — Inference Pipeline
- End-to-end OCR pipeline class usable from CLI or UI

### Phase 10 — Web App
- Original Streamlit research interface (kept for checkpoint-backed experiments)
- Production UI is now the FastAPI + React web application, deployed with Docker Compose

### Phase 11 — Tests
- pytest unit + integration tests

### Phase 12 — README & Docs

## Key Engineering Decisions

1. **Dataset**: IAM Handwriting Database lines split used as primary dataset. Synthetic printed text via text-renderer as secondary. Both configurable via `configs/dataset.yaml`.
2. **CTC decoding**: Greedy + optional beam search via `ctcdecode` or pure-Python fallback.
3. **Transformer OCR**: TrOCR (microsoft/trocr-base-handwritten) via HuggingFace — fine-tunable, documented clearly.
4. **PDF support**: `pymupdf` (fitz) for PDF→image conversion, optional dependency.
5. **Spell checking**: `pyspellchecker` (pure Python, no large model needed).
6. **Config**: `PyYAML` (already in most envs), no OmegaConf to keep deps minimal.
7. **No hardcoded paths**: All paths relative to project root via `pathlib`.
8. **Production OCR**: Pretrained PaddleOCR provides the web service's English detection/recognition engine; model weights are cached in a dedicated Docker volume.
9. **Upload privacy**: Web requests are bounded and transient. Only model weights persist; document files and OCR output are not retained.
