"""Checkpoint save/load helpers."""
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
    best_filename: str = "best_model.pth",
) -> Path:
    """Save a training checkpoint dict to disk."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    torch.save(state, path)
    logger.info(f"Checkpoint saved: {path}")

    if is_best:
        best_path = checkpoint_dir / best_filename
        shutil.copyfile(path, best_path)
        logger.info(f"Best model updated: {best_path}")

    return path


def load_checkpoint(
    checkpoint_path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint into a model (and optionally optimizer)."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Model weights loaded from {checkpoint_path}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state restored.")

    return checkpoint
