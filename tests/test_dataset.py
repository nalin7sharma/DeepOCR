"""Tests for dataset loading and augmentation."""
import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import cv2
from datasets.ocr_dataset import OCRDataset, build_vocabulary
from datasets.augmentations import OCRAugmentor


CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "
VOCAB = build_vocabulary(CHARSET)


def make_synthetic_dataset(tmp_path: Path, n: int = 5):
    """Create a minimal synthetic dataset in a temp dir."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    labels_path = tmp_path / "labels.txt"

    with open(labels_path, "w", encoding="utf-8") as f:
        for i in range(n):
            img = np.ones((64, 256), dtype=np.uint8) * 200
            img_name = f"img_{i:03d}.png"
            cv2.imwrite(str(img_dir / img_name), img)
            f.write("images/" + img_name + "\thello world " + str(i) + "\n")

    return str(tmp_path), str(labels_path)


class TestBuildVocabulary:
    def test_blank_at_index_0(self):
        chars, c2i, i2c = build_vocabulary("abc")
        assert chars[0] == "<BLANK>"
        assert c2i["<BLANK>"] == 0

    def test_all_chars_present(self):
        chars, c2i, i2c = build_vocabulary("abc")
        for ch in "abc":
            assert ch in c2i

    def test_round_trip(self):
        chars, c2i, i2c = build_vocabulary("hello")
        for ch in "hello":
            assert i2c[c2i[ch]] == ch


class TestOCRDataset:
    def test_load_simple_dataset(self, tmp_path):
        data_dir, gt_file = make_synthetic_dataset(tmp_path, n=5)
        ds = OCRDataset(data_dir, gt_file, VOCAB, 64, 256,
                        None, "train", "simple", 128)
        assert len(ds) == 5

    def test_getitem_shapes(self, tmp_path):
        data_dir, gt_file = make_synthetic_dataset(tmp_path, n=3)
        ds = OCRDataset(data_dir, gt_file, VOCAB, 64, 256,
                        None, "train", "simple", 128)
        sample = ds[0]
        assert sample["image"].shape == (1, 64, 256)
        assert sample["label"].dim() == 1
        assert isinstance(sample["label_text"], str)

    def test_label_encoding(self, tmp_path):
        data_dir, gt_file = make_synthetic_dataset(tmp_path, n=2)
        ds = OCRDataset(data_dir, gt_file, VOCAB, 64, 256,
                        None, "train", "simple", 128)
        sample = ds[0]
        # All label indices should be > 0 (0 is BLANK, not in output text)
        assert (sample["label"] > 0).all()


class TestAugmentor:
    def test_output_dtype(self):
        aug = OCRAugmentor()
        img = np.random.randint(0, 255, (64, 256), dtype=np.uint8)
        out = aug(img)
        assert out.dtype == np.uint8

    def test_output_channels_preserved(self):
        aug = OCRAugmentor()
        img = np.random.randint(0, 255, (64, 256, 3), dtype=np.uint8)
        out = aug(img)
        # Rotation enlarges the canvas instead of cropping, so only channels are fixed.
        assert out.ndim == 3 and out.shape[2] == 3
        assert out.shape[0] >= 32 and out.shape[1] >= 128

    def test_scale_never_crops_ink(self):
        aug = OCRAugmentor({"scale_range": [1.2, 1.2]})
        img = np.zeros((32, 200), dtype=np.uint8)
        img[:, -5:] = 255  # ink at the far right edge
        out = aug._random_scale(img)
        assert out.shape[1] > img.shape[1]
        assert out[:, -3:].max() == 255

    def test_dataset_normalises_polarity(self, tmp_path):
        data_dir, gt_file = make_synthetic_dataset(tmp_path, n=1)
        img = np.full((64, 256), 230, dtype=np.uint8)
        cv2.putText(img, "Hi", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 10, 3)
        cv2.imwrite(str(Path(data_dir) / "images" / "img_000.png"), img)
        ds = OCRDataset(data_dir, gt_file, VOCAB, 32, 256, None, "test", "simple", 128)
        tensor = ds[0]["image"]
        assert tensor.shape == (1, 32, 256)
        # Background maps to ~0 and ink to ~1 after normalisation
        assert float(tensor.median()) < 0.05
        assert float(tensor.max()) > 0.9
