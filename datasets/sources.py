"""
Dataset source resolution shared by training and evaluation.

A *source* is a named dataset directory (``iam``, ``synthetic`` or any path)
that either provides explicit split manifests (``train.txt``, ``val.txt``,
``test.txt``) or a single legacy ``labels.txt`` that is partitioned with the
deterministic ``split_dataset_indices`` helper.
"""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import ConcatDataset, Dataset, Subset

from .ocr_dataset import OCRDataset
from .splits import split_dataset_indices

SPLIT_FILES = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}


def source_dir(name: str, root: Path, cfg: dict) -> Path:
    """Map a source name to its directory (config ``<name>.processed_dir`` or a path)."""
    section = cfg.get(name) or cfg.get("dataset", {}).get(name) or {}
    configured = section.get("processed_dir") if isinstance(section, dict) else None
    if configured:
        return root / configured
    candidate = Path(name)
    return candidate if candidate.is_absolute() else root / candidate


def load_split(
    name: str,
    split: str,
    vocab: tuple,
    root: Path,
    cfg: dict,
    augmentor=None,
) -> Dataset:
    """Return the ``split`` partition of one dataset source."""
    dataset_cfg = cfg.get("dataset", {})
    height = dataset_cfg.get("image_height", 32)
    width = dataset_cfg.get("image_width", 512)
    max_label = dataset_cfg.get("max_label_length", 96)
    directory = source_dir(name, root, cfg)
    mode = "train" if split == "train" else split

    manifest = directory / SPLIT_FILES[split]
    if manifest.exists():
        return OCRDataset(directory, manifest, vocab, height, width,
                          augmentor if split == "train" else None, mode, "simple", max_label)

    legacy = directory / "labels.txt"
    if not legacy.exists():
        raise FileNotFoundError(
            f"No {SPLIT_FILES[split]} or labels.txt in {directory}. "
            "Prepare data: python scripts/download_iam_hf.py / scripts/prepare_dataset.py"
        )
    dataset = OCRDataset(directory, legacy, vocab, height, width,
                         augmentor if split == "train" else None, mode, "auto", max_label)
    fractions = cfg.get("splits", {"train": 0.8, "val": 0.1, "test": 0.1})
    seed = cfg.get("training", {}).get("seed", 42)
    indices = split_dataset_indices(len(dataset), fractions, seed)
    return Subset(dataset, indices[split])


def load_sources(names: list[str], split: str, vocab: tuple, root: Path, cfg: dict,
                 augmentor=None) -> Dataset:
    parts = [load_split(name, split, vocab, root, cfg, augmentor) for name in names]
    parts = [part for part in parts if len(part) > 0]
    if not parts:
        raise ValueError(f"No {split} samples found in sources {names}.")
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)
