"""Integration test: OCR pipeline (no checkpoint required)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from inference.ocr_pipeline import OCRPipeline
from datasets.ocr_dataset import build_vocabulary
from models.cnn_bilstm import build_cnn_bilstm
from utils.config_loader import load_all_configs


@pytest.fixture
def blank_image():
    """512x128 white BGR image."""
    return np.ones((128, 512, 3), dtype=np.uint8) * 255


class TestOCRPipeline:
    @staticmethod
    def _untrained_pipeline():
        """Build an in-memory model for pipeline mechanics tests only."""
        cfg = load_all_configs()
        vocab = build_vocabulary(cfg["dataset"]["charset"])
        model = build_cnn_bilstm(len(vocab[0]), cfg.get("cnn_bilstm", {}))
        return OCRPipeline(model=model, vocab=vocab)

    def test_from_config_requires_trained_checkpoint(self, tmp_path):
        """The production pipeline must not expose random-weight predictions."""
        with pytest.raises(FileNotFoundError, match="No trained checkpoint"):
            OCRPipeline.from_config(model_name="cnn_bilstm", checkpoint_path=tmp_path / "missing.pth",
                                    detector="opencv")

    def test_single_column_lines_keep_reading_order(self):
        from layout.layout_analyzer import LayoutAnalyzer
        # Ragged left-aligned lines of different lengths are one column, not several.
        detections = [{"bbox": (40, 20 + i * 40, width, 25)} for i, width in enumerate([500, 180, 420, 90, 300])]
        blocks = LayoutAnalyzer().analyze(detections, image_width=600, image_height=260)
        assert [b.bbox[1] for b in blocks] == [20, 60, 100, 140, 180]
        assert {b.column for b in blocks} == {0}

    def test_two_columns_are_read_column_by_column(self):
        from layout.layout_analyzer import LayoutAnalyzer
        left = [{"bbox": (20, 50 + i * 30, 250, 20)} for i in range(3)]
        right = [{"bbox": (330, 50 + i * 30, 250, 20)} for i in range(3)]
        title = [{"bbox": (20, 10, 560, 25)}]
        blocks = LayoutAnalyzer().analyze(right + left + title, image_width=600, image_height=200)
        assert [b.bbox[0] for b in blocks] == [20, 20, 20, 20, 330, 330, 330]

    def test_run_returns_dict(self, blank_image):
        pipeline = self._untrained_pipeline()
        result = pipeline.run(blank_image)
        assert isinstance(result, dict)
        assert "text" in result
        assert "inference_ms" in result
        assert "regions" in result
        assert result["checkpoint_loaded"] is False

    def test_inference_time_positive(self, blank_image):
        pipeline = self._untrained_pipeline()
        result = pipeline.run(blank_image)
        assert result["inference_ms"] > 0

    def test_run_from_pil(self, blank_image):
        from PIL import Image
        pil_img = Image.fromarray(blank_image, mode="RGB")
        pipeline = self._untrained_pipeline()
        result = pipeline.run(pil_img)
        assert "text" in result
