#!/usr/bin/env python
"""
Real-world robustness test: CER on IAM test lines under controlled degradation.

For a fixed random subset of IAM *test* lines, each condition is applied at
increasing severity and every model re-reads the degraded lines:

  noise     additive Gaussian noise, sigma (grey levels)
  blur      Gaussian defocus blur, sigma (pixels at 128 px line height)
  rotation  in-plane rotation, degrees
  rotation_deskew  the same rotation, then DeepOCR's projection-profile deskew
  contrast  contrast reduction, remaining contrast factor
  jpeg      JPEG compression, quality

Outputs results/evaluation/robustness.json (read by the dashboard) and
results/plots/robustness.png (for the report).

Usage:
    python scripts/robustness.py                       # all models, 120 lines
    python scripts/robustness.py --models crnn cnn_bilstm --lines 200
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.line_image import normalize_transcription
from evaluation.metrics import cer_score
from scripts.evaluate import MODEL_INFO, build_recognizer, load_test_set
from utils.config_loader import get_project_root, load_all_configs
from utils.logger import get_logger

logger = get_logger(__name__)

CONDITIONS = [
    {"id": "noise", "label": "Gaussian noise (σ)", "levels": [0, 15, 30, 50]},
    {"id": "blur", "label": "Blur (σ px)", "levels": [0, 1.0, 2.0, 3.0]},
    {"id": "rotation", "label": "Rotation (°)", "levels": [0, 2, 5, 8]},
    {"id": "rotation_deskew", "label": "Rotation + our deskew (°)", "levels": [0, 2, 5, 8]},
    {"id": "contrast", "label": "Contrast remaining", "levels": [1.0, 0.5, 0.3, 0.15]},
    {"id": "jpeg", "label": "JPEG quality", "levels": [100, 40, 20, 8]},
]


def crop_to_ink(image: np.ndarray, margin: float = 0.12) -> np.ndarray:
    """Tight crop around the ink, as a text detector would deliver after deskewing."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ys, xs = np.nonzero(ink)
    if ys.size < 20:
        return image
    y0, y1 = np.percentile(ys, [0.5, 99.5]).astype(int)
    x0, x1 = np.percentile(xs, [0.2, 99.8]).astype(int)
    pad = int((y1 - y0) * margin) + 2
    return image[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]


def degrade(image: np.ndarray, condition: str, level: float, seed: int) -> np.ndarray:
    img = image.copy()
    if condition == "noise" and level:
        rng = np.random.default_rng(seed)
        img = np.clip(img.astype(np.float32) + rng.normal(0, level, img.shape), 0, 255).astype(np.uint8)
    elif condition == "blur" and level:
        img = cv2.GaussianBlur(img, (0, 0), level)
    elif condition == "rotation_deskew" and level:
        from api.preprocess import estimate_skew, rotate_page

        rotated = degrade(image, "rotation", level, seed)
        angle = estimate_skew(rotated)
        img = crop_to_ink(rotate_page(rotated, angle)) if abs(angle) >= 0.3 else rotated
    elif condition == "rotation" and level:
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), level, 1.0)
        cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
        new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
        matrix[0, 2] += new_w / 2 - w / 2
        matrix[1, 2] += new_h / 2 - h / 2
        img = cv2.warpAffine(img, matrix, (new_w, new_h), borderValue=(255, 255, 255))
    elif condition == "contrast" and level < 1.0:
        mean = float(img.mean())
        img = np.clip(mean + (img.astype(np.float32) - mean) * level, 0, 255).astype(np.uint8)
    elif condition == "jpeg" and level < 100:
        ok, data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(level)])
        img = cv2.imdecode(data, cv2.IMREAD_COLOR) if ok else img
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["cnn_bilstm", "crnn", "trocr", "paddle"])
    parser.add_argument("--lines", type=int, default=120)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = load_all_configs()
    root = get_project_root()
    rows = load_test_set(root, cfg, "iam", None)
    rows = random.Random(11).sample(rows, min(args.lines, len(rows)))
    images = [cv2.imread(str(path)) for path, _ in rows]
    refs = [normalize_transcription(text) for _, text in rows]

    out_path = root / "results" / "evaluation" / "robustness.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    payload.update({"conditions": CONDITIONS, "samples": len(rows)})
    payload.setdefault("models", {})

    for model_name in args.models:
        try:
            recognizer = build_recognizer(model_name, root, cfg)
        except Exception as exc:  # noqa: BLE001 - report and continue with other models
            logger.error(f"Skipping {model_name}: {exc}")
            continue
        clean_cer = None
        curves: dict[str, list[float]] = {}
        for condition in CONDITIONS:
            values = []
            for index, level in enumerate(condition["levels"]):
                if index == 0 and clean_cer is not None:
                    values.append(clean_cer)  # level 0 is the clean image for every condition
                    continue
                degraded = [degrade(img, condition["id"], level, seed=i) for i, img in enumerate(images)]
                preds = [normalize_transcription(text) for text, _ in recognizer.recognize(degraded)]
                cer = 100 * sum(cer_score(p, r) for p, r in zip(preds, refs)) / len(refs)
                values.append(round(cer, 2))
                if index == 0:
                    clean_cer = values[0]
                logger.info(f"{model_name} {condition['id']}={level}: CER {cer:.2f}%")
            curves[condition["id"]] = values
        payload["models"][model_name] = curves
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        del recognizer

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        palette = {"cnn_bilstm": "#06b6d4", "crnn": "#8b5cf6", "trocr": "#65a30d", "trocr_ft": "#ca8a04", "paddle": "#64748b"}
        fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(4.2 * len(CONDITIONS), 3.6))
        for ax, condition in zip(axes, CONDITIONS):
            for model_name, curves in payload["models"].items():
                if condition["id"] in curves:
                    ax.plot(range(len(condition["levels"])), curves[condition["id"]], marker="o",
                            label=MODEL_INFO.get(model_name, {}).get("label", model_name), color=palette.get(model_name))
            ax.set_xticks(range(len(condition["levels"])), [str(level) for level in condition["levels"]])
            ax.set_title(condition["label"])
            ax.set_ylabel("CER %")
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        (root / "results" / "plots").mkdir(parents=True, exist_ok=True)
        fig.savefig(root / "results" / "plots" / "robustness.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Plot failed: {exc}")
    print(json.dumps(payload["models"], indent=2))


if __name__ == "__main__":
    main()
