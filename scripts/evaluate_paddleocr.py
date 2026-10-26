#!/usr/bin/env python
"""Measure the production PaddleOCR engine against a labeled TSV manifest."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.ocr_engine import OcrEngine, OcrEngineError, PaddleOcrEngine
from evaluation.metrics import (
    batch_cer,
    batch_wer,
    recognition_accuracy,
    cer_score,
    wer_score,
)


def load_manifest(manifest_path: Path, images_dir: Path) -> list[tuple[Path, str]]:
    """Read tab-separated relative image paths and exact reference transcriptions."""
    root = images_dir.resolve()
    samples = []
    with manifest_path.open(encoding="utf-8-sig", newline="") as manifest_file:
        for line_number, row in enumerate(csv.reader(manifest_file, delimiter="\t"), start=1):
            if not row or (row[0].lstrip().startswith("#")):
                continue
            if len(row) == 2 and row[0].strip().lower() in {
                "image", "image_path", "filename", "path"
            }:
                continue
            if len(row) != 2 or not row[0].strip():
                raise ValueError(
                    f"{manifest_path}:{line_number}: expected image_path<TAB>ground_truth."
                )
            relative_path = Path(row[0])
            if relative_path.is_absolute():
                raise ValueError(
                    f"{manifest_path}:{line_number}: image paths must be relative to {root}."
                )
            image_path = (root / relative_path).resolve()
            try:
                image_path.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"{manifest_path}:{line_number}: image path escapes the image directory."
                ) from exc
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"{manifest_path}:{line_number}: image not found: {image_path}"
                )
            samples.append((image_path, row[1]))
    if not samples:
        raise ValueError(f"No labeled samples found in {manifest_path}.")
    return samples


def evaluate_samples(
    samples: list[tuple[Path, str]],
    engine: OcrEngine,
    language: str,
) -> dict:
    """Run OCR and calculate per-sample macro CER/WER and exact-match accuracy."""
    predictions = []
    references = []
    elapsed_seconds = 0.0
    for image_path, reference in samples:
        with Image.open(image_path) as source:
            image_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        started = time.perf_counter()
        regions = engine.recognize(image_bgr)
        elapsed_seconds += time.perf_counter() - started
        prediction = " ".join(region.text.strip() for region in regions if region.text.strip())
        predictions.append(prediction)
        references.append(reference)

    details = [
        {
            "image": str(image_path),
            "pred": prediction,
            "gt": reference,
            "cer": round(cer_score(prediction, reference) * 100, 4),
            "wer": round(wer_score(prediction, reference) * 100, 4),
        }
        for (image_path, _), prediction, reference in zip(
            samples, predictions, references
        )
    ]
    return {
        "model": "PaddleOCR",
        "language": language,
        "num_samples": len(samples),
        "cer": round(batch_cer(predictions, references) * 100, 4),
        "wer": round(batch_wer(predictions, references) * 100, 4),
        "accuracy": round(recognition_accuracy(predictions, references) * 100, 4),
        "inference_ms": round(elapsed_seconds * 1000 / len(samples), 2),
        "predictions": details,
    }


def save_results(results: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "paddleocr_results.json"
    csv_path = output_dir / "paddleocr_predictions.csv"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2, ensure_ascii=False)
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=("image", "pred", "gt", "cer", "wer")
        )
        writer.writeheader()
        writer.writerows(results["predictions"])
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PaddleOCR against a labeled image-transcription manifest."
    )
    parser.add_argument("--manifest", required=True, type=Path,
                        help="UTF-8 TSV: relative_image_path<TAB>ground_truth")
    parser.add_argument("--images-dir", required=True, type=Path,
                        help="Root directory for paths in the manifest")
    parser.add_argument("--language", default="en")
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluations"))
    args = parser.parse_args()

    try:
        samples = load_manifest(args.manifest, args.images_dir)
        results = evaluate_samples(
            samples,
            PaddleOcrEngine(language=args.language),
            args.language,
        )
    except (FileNotFoundError, OcrEngineError, OSError, ValueError) as exc:
        parser.error(str(exc))
    json_path, csv_path = save_results(results, args.output_dir)
    print(
        f"PaddleOCR | samples={results['num_samples']} "
        f"CER={results['cer']:.2f}% WER={results['wer']:.2f}% "
        f"exact-match accuracy={results['accuracy']:.2f}% "
        f"inference={results['inference_ms']:.1f} ms/sample"
    )
    print(f"Results: {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
