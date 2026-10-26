"""
General OCR Dataset for DeepOCR.

Supports IAM line-level dataset format and a simple folder-based format:
  <data_dir>/
      images/   *.png
      labels.txt   (filename\tground_truth per line)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import OCRAugmentor
from .base_dataset import BaseOCRDataset
from .line_image import fit_width, normalize_line, normalize_transcription, resize_to_height, to_tensor


# Special token indices
BLANK_IDX = 0  # CTC blank


def build_vocabulary(charset: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Build character vocabulary from a charset string.

    Returns:
        chars: Ordered list starting with BLANK at index 0.
        char2idx: Character to index mapping.
        idx2char: Index to character mapping.
    """
    unique_chars = sorted(set(charset))
    chars = ["<BLANK>"] + unique_chars  # index 0 = CTC blank
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}
    return chars, char2idx, idx2char


class OCRDataset(BaseOCRDataset):
    """Dataset that loads (image, text_label) pairs for OCR training.

    Supports two file layouts:
      1. IAM-lines format  (gt_file = lines.txt with IAM annotation format)
      2. Simple format     (gt_file = labels.txt with "image_path\ttext" lines)

    Args:
        data_dir: Root directory of the dataset.
        gt_file: Path to the ground-truth annotation file.
        vocab: Vocabulary tuple (chars, char2idx, idx2char).
        image_height: Fixed height images are scaled to.
        image_width: Fixed width (with padding).
        augmentor: Optional augmentation callable.
        mode: "train" | "val" | "test" — controls augmentation.
        dataset_type: "simple" | "iam"
    """

    def __init__(
        self,
        data_dir: str | Path,
        gt_file: str | Path,
        vocab: tuple[list[str], dict[str, int], dict[int, str]],
        image_height: int = 64,
        image_width: int = 512,
        augmentor: OCRAugmentor | None = None,
        mode: str = "train",
        dataset_type: str = "auto",
        max_label_length: int = 128,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.gt_file = Path(gt_file)
        self.chars, self.char2idx, self.idx2char = vocab
        self.image_height = image_height
        self.image_width = image_width
        self.augmentor = augmentor
        self.mode = mode
        self.max_label_length = max_label_length
        self.dataset_type = dataset_type

        self.samples: list[dict] = []  # list of {"image_path", "label"}
        self._load_samples()

    # ------------------------------------------------------------------
    # BaseOCRDataset interface
    # ------------------------------------------------------------------

    @property
    def vocabulary(self) -> list[str]:
        return self.chars

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = self._load_image(sample["image_path"])
        label_text = sample["label"]

        # Shared normalisation: bright ink on black, fixed height (see line_image.py)
        image = resize_to_height(normalize_line(image), self.image_height)

        # Augmentation only during training
        if self.mode == "train" and self.augmentor is not None:
            image = resize_to_height(self.augmentor(image), self.image_height)

        # Convert to float tensor [1, H, W] normalised to [0, 1]
        image_tensor = to_tensor(fit_width(image, self.image_width))

        # Encode label to integer sequence
        label_encoded = self._encode_label(label_text)
        label_tensor = torch.tensor(label_encoded, dtype=torch.long)

        return {
            "image": image_tensor,
            "label": label_tensor,
            "label_text": label_text,
            "label_length": torch.tensor(len(label_encoded), dtype=torch.long),
            "image_path": str(sample["image_path"]),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_samples(self) -> None:
        """Parse ground-truth file and build self.samples list."""
        if not self.gt_file.exists():
            raise FileNotFoundError(f"Ground-truth file not found: {self.gt_file}")

        dtype = self.dataset_type
        if dtype == "auto":
            # Auto-detect format by inspecting the first non-comment line
            dtype = "simple"
            with open(self.gt_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "\t" in line:
                        dtype = "simple"
                    elif "ok" in line or "err" in line:
                        dtype = "iam"
                    break

        if dtype == "iam":
            self._load_iam_samples()
        else:
            self._load_simple_samples()

    def _load_simple_samples(self) -> None:
        """Load simple tab-separated format: relative_image_path\tground_truth."""
        with open(self.gt_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                img_rel, label = parts
                img_path = self.data_dir / img_rel
                # Filter out-of-vocab characters
                label = self._filter_label(label)
                if label and img_path.exists():
                    self.samples.append({"image_path": img_path, "label": label})

    def _load_iam_samples(self) -> None:
        """Load IAM lines.txt format."""
        with open(self.gt_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split(" ")
                if len(parts) < 9:
                    continue
                line_id = parts[0]           # e.g. a01-000u-00
                ok_flag = parts[1]           # ok | err
                if ok_flag == "err":
                    continue
                # Transcription is the last space-joined field
                # IAM format: line_id ok gray_val components_x_y_w_h num_components transcription
                transcription = " ".join(parts[8:]).replace("|", " ")
                transcription = self._filter_label(transcription)
                if not transcription:
                    continue

                # Construct image path from IAM directory structure:
                # a01-000u-00 -> lines/a01/a01-000u/a01-000u-00.png
                parts_id = line_id.split("-")
                if len(parts_id) < 3:
                    continue
                writer = parts_id[0]
                form = f"{parts_id[0]}-{parts_id[1]}"
                filename = f"{line_id}.png"
                img_path = self.data_dir / "lines" / writer / form / filename

                if img_path.exists():
                    self.samples.append({"image_path": img_path, "label": transcription})

    def _filter_label(self, label: str) -> str:
        """Normalise spacing, then remove characters not in vocabulary."""
        label = normalize_transcription(label)
        filtered = "".join(c for c in label if c in self.char2idx)
        if len(filtered) > self.max_label_length:
            filtered = filtered[: self.max_label_length]
        return filtered

    def _encode_label(self, label: str) -> list[int]:
        """Convert text string to list of vocab indices (excluding BLANK)."""
        return [self.char2idx[c] for c in label if c in self.char2idx]

    def _load_image(self, path: Path) -> np.ndarray:
        """Load an image as grayscale numpy array."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Return blank image if file is corrupt / missing
            img = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        return img

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Resize, normalise and convert grayscale image to [1, H, W] float tensor."""
        # Resize to target dimensions
        h, w = image.shape[:2]
        scale = self.image_height / h
        new_w = min(int(w * scale), self.image_width)
        resized = cv2.resize(image, (new_w, self.image_height))

        # Pad to target_width
        canvas = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        canvas[:, :new_w] = resized

        # Normalise to [0, 1] and add channel dimension
        tensor = torch.from_numpy(canvas).float() / 255.0
        return tensor.unsqueeze(0)  # [1, H, W]
