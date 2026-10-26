#!/usr/bin/env python
"""
Standalone Inference Script for DeepOCR.

Usage:
    python scripts/infer.py --image path/to/image.png --model cnn_bilstm
    python scripts/infer.py --image doc.jpg --model crnn --output result.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.ocr_pipeline import OCRPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR on an image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="cnn_bilstm",
                        choices=["cnn_bilstm", "crnn", "transformer_ocr"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Optional output text file path")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)

    logger.info(f"Loading pipeline for model: {args.model}")
    try:
        pipeline = OCRPipeline.from_config(
            model_name=args.model,
            checkpoint_path=args.checkpoint,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info(f"Running OCR on: {image_path}")
    result = pipeline.run(str(image_path))

    print("\n" + "="*60)
    print("EXTRACTED TEXT:")
    print("="*60)
    print(result["text"])
    print("="*60)
    print(f"Inference time: {result['inference_ms']:.1f} ms")
    print(f"Regions detected: {len(result['regions'])}")

    if args.output:
        Path(args.output).write_text(result["text"], encoding="utf-8")
        logger.info(f"Text saved to: {args.output}")


if __name__ == "__main__":
    main()
