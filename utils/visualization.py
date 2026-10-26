"""Visualisation helpers: bounding boxes, metrics plots, etc."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_bboxes(
    image: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    labels: list[str] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on an image (BGR numpy array).

    Args:
        image: BGR image as numpy array.
        bboxes: List of (x, y, w, h) tuples.
        labels: Optional text label per box.
        color: Box colour (BGR).
        thickness: Line thickness.

    Returns:
        Annotated image copy.
    """
    img = image.copy()
    for i, (x, y, w, h) in enumerate(bboxes):
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        if labels and i < len(labels):
            cv2.putText(img, labels[i], (x, max(y - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def plot_training_curves(
    history: dict[str, list[float]],
    output_path: str | Path,
    title: str = "Training Curves",
) -> None:
    """Save a training/validation loss + metric plot.

    Args:
        history: Dict with keys like "train_loss", "val_cer", etc.
        output_path: Where to save the PNG.
        title: Plot title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax = axes[0]
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val Loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # CER / WER
    ax = axes[1]
    for key in ("train_cer", "val_cer", "val_wer"):
        if key in history:
            ax.plot(history[key], label=key.replace("_", " ").title())
    ax.set_title("Error Rates")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def save_comparison_table(
    results: list[dict],
    output_path: str | Path,
    title: str = "Model Comparison",
) -> None:
    """Save a model-comparison table as a PNG figure.

    Args:
        results: List of dicts with keys: model, cer, wer, accuracy, inference_ms.
        output_path: Where to save.
        title: Figure title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = ["Model", "CER (%)", "WER (%)", "Accuracy (%)", "Inference (ms)"]
    cell_text = []
    for r in results:
        row = [
            r.get("model", "—"),
            f"{r.get('cer', float('nan')):>6.2f}" if r.get("cer") is not None else "Not evaluated",
            f"{r.get('wer', float('nan')):>6.2f}" if r.get("wer") is not None else "Not evaluated",
            f"{r.get('accuracy', float('nan')):>6.2f}" if r.get("accuracy") is not None else "Not evaluated",
            f"{r.get('inference_ms', float('nan')):>6.1f}" if r.get("inference_ms") is not None else "—",
        ]
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(10, max(2, len(results) + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)
    ax.set_title(title, fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
