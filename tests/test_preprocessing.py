"""Unit tests for image preprocessing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from preprocessing.image_processor import ImageProcessor


@pytest.fixture
def sample_gray():
    """128x512 grayscale uint8 image."""
    return np.random.randint(50, 200, (128, 512), dtype=np.uint8)


@pytest.fixture
def sample_bgr():
    """128x512 BGR uint8 image."""
    return np.random.randint(50, 200, (128, 512, 3), dtype=np.uint8)


class TestImageProcessor:
    def test_to_grayscale_from_bgr(self, sample_bgr):
        gray = ImageProcessor.to_grayscale(sample_bgr)
        assert gray.ndim == 2

    def test_to_grayscale_noop_on_gray(self, sample_gray):
        gray = ImageProcessor.to_grayscale(sample_gray)
        assert gray.shape == sample_gray.shape

    def test_normalize_output_range(self, sample_gray):
        norm = ImageProcessor.normalize_brightness(sample_gray)
        assert norm.min() >= 0
        assert norm.max() <= 255
        assert norm.dtype == np.uint8

    def test_sharpen_shape_preserved(self, sample_gray):
        sharp = ImageProcessor.sharpen(sample_gray)
        assert sharp.shape == sample_gray.shape

    def test_resize_and_pad_output_shape(self, sample_bgr):
        result = ImageProcessor.resize_and_pad(sample_bgr, target_height=64, target_width=512)
        assert result.shape[0] == 64
        assert result.shape[1] == 512

    def test_resize_gray_output_shape(self, sample_gray):
        result = ImageProcessor.resize_and_pad(sample_gray, target_height=64, target_width=512)
        assert result.shape == (64, 512)

    def test_process_pipeline(self, sample_bgr):
        proc = ImageProcessor()
        result = proc.process(sample_bgr)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8

    def test_binarize_binary_output(self, sample_gray):
        binary = ImageProcessor.binarize(sample_gray)
        unique = np.unique(binary)
        assert set(unique).issubset({0, 255})

    def test_enhance_contrast_gray(self, sample_gray):
        enhanced = ImageProcessor.enhance_contrast(sample_gray)
        assert enhanced.shape == sample_gray.shape
