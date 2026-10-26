#!/usr/bin/env python
"""
Model Comparison Script for DeepOCR.

Loads evaluation results for all models that have been evaluated and
generates a comparison table (CSV + PNG).

If a model has not yet been evaluated, it shows "Not evaluated yet"
rather than inventing numbers.

Usage:
    python scripts/compare_models.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_loader import get_project_root
from utils.logger import get_logger
from utils.visualization import save_comparison_table

logger = get_logger(__name__)

MODELS = ["cnn_bilstm", "crnn", "transformer_ocr"]
DISPLAY_NAMES = {
    "cnn_bilstm": "CNN + BiLSTM (Baseline)",
    "crnn": "CRNN (Shi et al.)",
    "transformer_ocr": "Transformer OCR (Advanced)",
}


def load_result(results_dir: Path, model_name: str) -> dict | None:
    """Load evaluation result JSON if it exists."""
    result_path = results_dir / "evaluations" / f"{model_name}_results.json"
    if result_path.exists():
        with open(result_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def main() -> None:
    root = get_project_root()
    results_dir = root / "results"

    rows = []
    for model_name in MODELS:
        result = load_result(results_dir, model_name)
        if result:
            rows.append({
                "model": DISPLAY_NAMES[model_name],
                "cer": result.get("cer"),
                "wer": result.get("wer"),
                "accuracy": result.get("accuracy"),
                "inference_ms": result.get("inference_ms"),
                "num_samples": result.get("num_samples", "-"),
            })
        else:
            rows.append({
                "model": DISPLAY_NAMES[model_name],
                "cer": None,
                "wer": None,
                "accuracy": None,
                "inference_ms": None,
            })

    # Print to console
    print("\n" + "="*70)
    print(f"{'MODEL COMPARISON':^70}")
    print(f"{'='*70}")
    header = f"{'Model':<35} {'CER%':>8} {'WER%':>8} {'Acc%':>8} {'Inf(ms)':>10}"
    print(header)
    print("-" * 70)
    for r in rows:
        cer = f"{r['cer']:>8.2f}" if r["cer"] is not None else "  Not evaluated"
        wer = f"{r['wer']:>8.2f}" if r["wer"] is not None else "  Not evaluated"
        acc = f"{r['accuracy']:>8.2f}" if r["accuracy"] is not None else "  Not evaluated"
        inf = f"{r['inference_ms']:>10.1f}" if r["inference_ms"] is not None else "         -"
        print(f"{r['model']:<35}{cer}{wer}{acc}{inf}")
    print("=" * 70)

    # Save CSV
    import csv
    csv_path = results_dir / "model_comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "cer", "wer", "accuracy", "inference_ms"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "model": r["model"],
                "cer": r["cer"] if r["cer"] is not None else "Not evaluated",
                "wer": r["wer"] if r["wer"] is not None else "Not evaluated",
                "accuracy": r["accuracy"] if r["accuracy"] is not None else "Not evaluated",
                "inference_ms": r["inference_ms"] if r["inference_ms"] is not None else "-",
            })
    logger.info(f"Comparison CSV saved: {csv_path}")

    # Save comparison table PNG
    png_path = results_dir / "plots" / "model_comparison.png"
    save_comparison_table(rows, png_path)
    logger.info(f"Comparison table PNG saved: {png_path}")


if __name__ == "__main__":
    main()
