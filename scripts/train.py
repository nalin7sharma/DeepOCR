#!/usr/bin/env python
"""
Training Script for DeepOCR CTC models (CNN-BiLSTM baseline and CRNN).

Data: the sources listed in ``configs/dataset.yaml -> dataset.train_sets``
(default: IAM handwriting + synthetic printed/handwriting-font lines), using
each source's explicit train/val partitions.  The test partitions are never
touched here — see ``scripts/evaluate.py``.

Usage:
    python scripts/train.py --model cnn_bilstm
    python scripts/train.py --model crnn --epochs 25 --max_minutes 120
    python scripts/train.py --model crnn --datasets iam          # handwriting only

The advanced Transformer model is pretrained TrOCR used via transfer learning;
evaluate it with ``python scripts/evaluate.py --model trocr``.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.augmentations import OCRAugmentor
from datasets.ocr_dataset import build_vocabulary
from datasets.sources import load_sources
from models.cnn_bilstm import build_cnn_bilstm
from models.crnn import build_crnn
from training.trainer import Trainer
from utils.config_loader import get_project_root, load_all_configs
from utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_BUILDERS = {
    "cnn_bilstm": build_cnn_bilstm,
    "crnn": build_crnn,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """Custom collate: pad labels to same length within batch."""
    images = torch.stack([b["image"] for b in batch])
    label_lengths = torch.stack([b["label_length"] for b in batch])
    max_len = label_lengths.max().item()
    labels_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        l = b["label"]
        labels_padded[i, : len(l)] = l
    label_texts = [b["label_text"] for b in batch]
    return {
        "image": images,
        "label": labels_padded,
        "label_length": label_lengths,
        "label_text": label_texts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DeepOCR CTC model")
    parser.add_argument("--model", default="cnn_bilstm", choices=list(_MODEL_BUILDERS),
                        help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_minutes", type=float, default=None,
                        help="Wall-clock budget; stops cleanly after the last epoch that fits")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset sources (names from dataset.yaml or directories)")
    parser.add_argument("--limit_train", type=int, default=None,
                        help="Use only the first N training samples (smoke tests)")
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads for PyTorch (useful when training models in parallel)")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader worker processes")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = load_all_configs()
    root = get_project_root()
    train_cfg = cfg.setdefault("training", {})
    dataset_cfg = cfg.get("dataset", {})
    aug_cfg = cfg.get("augmentation", {})

    for key in ("epochs", "batch_size", "seed", "max_minutes"):
        value = getattr(args, key)
        if value is not None:
            train_cfg[key] = value
    if args.lr is not None:
        train_cfg["learning_rate"] = args.lr
    train_cfg["model"] = args.model

    seed = train_cfg.get("seed", 42)
    set_seed(seed)
    sources = args.datasets or dataset_cfg.get("train_sets") or ["iam", "synthetic"]
    logger.info(f"Model: {args.model} | sources: {sources} | seed: {seed}")

    charset = dataset_cfg["charset"]
    vocab = build_vocabulary(charset)
    num_classes = len(vocab[0])
    logger.info(f"Vocabulary size: {num_classes} ({num_classes - 1} chars + BLANK)")

    augmentor = OCRAugmentor(aug_cfg) if aug_cfg.get("enabled", True) else None
    try:
        train_ds = load_sources(sources, "train", vocab, root, cfg, augmentor)
        val_ds = load_sources(sources, "val", vocab, root, cfg)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)
    if args.limit_train:
        train_ds = torch.utils.data.Subset(train_ds, list(range(min(args.limit_train, len(train_ds)))))
    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    batch_size = train_cfg.get("batch_size", 32)
    num_workers = args.workers if args.workers is not None else min(train_cfg.get("num_workers", 0), 4)
    loader_kwargs = {"num_workers": num_workers, "collate_fn": collate_fn,
                     "persistent_workers": num_workers > 0}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **loader_kwargs)

    image_height = dataset_cfg.get("image_height", 32)
    image_width = dataset_cfg.get("image_width", 512)
    model_cfg = dict(cfg.get(args.model, {}))
    model = _MODEL_BUILDERS[args.model](num_classes, model_cfg, image_height=image_height)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    meta = {
        "model_name": args.model,
        "model_cfg": model_cfg,
        "charset": charset,
        "image_height": image_height,
        "image_width": image_width,
        "datasets": sources,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "parameters": n_params,
        "seed": seed,
        "augmentation": aug_cfg if augmentor else None,
    }
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config={"training": train_cfg},
        checkpoint_dir=root / "checkpoints",
        results_dir=root / "results",
        vocab=vocab,
        model_type="ctc",
        meta=meta,
    )
    trainer.train()
    logger.info(f"Training complete. Best val CER: {trainer.best_metric:.4f}")


if __name__ == "__main__":
    main()
