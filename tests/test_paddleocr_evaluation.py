import csv

import numpy as np
import pytest
from PIL import Image

from api.ocr_engine import TextRegion
from scripts.evaluate_paddleocr import evaluate_samples, load_manifest, save_results


class FakeEngine:
    def recognize(self, image):
        assert image.shape == (20, 40, 3)
        assert image[0, 0].tolist() == [0, 0, 255]
        return [
            TextRegion(
                polygon=[[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]],
                text="hello",
                confidence=0.95,
            )
        ]


def test_loads_labeled_tsv_and_evaluates_real_sample_set(tmp_path):
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    Image.fromarray(image).save(tmp_path / "sample.png")
    manifest = tmp_path / "labels.tsv"
    manifest.write_text(
        "image_path\tground_truth\n# comment\nsample.png\thello\nsample.png\tworld\n",
        encoding="utf-8",
    )

    samples = load_manifest(manifest, tmp_path)
    results = evaluate_samples(samples, FakeEngine(), "en")

    assert results["model"] == "PaddleOCR"
    assert results["num_samples"] == 2
    assert results["accuracy"] == 50.0
    assert results["cer"] > 0
    assert results["predictions"][0]["pred"] == "hello"

    json_path, csv_path = save_results(results, tmp_path / "results")
    assert json_path.is_file()
    with csv_path.open(encoding="utf-8", newline="") as predictions_file:
        rows = list(csv.DictReader(predictions_file))
    assert len(rows) == 2
    assert rows[0]["gt"] == "hello"


def test_manifest_rejects_paths_outside_image_root(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    manifest = tmp_path / "labels.tsv"
    manifest.write_text("../outside.png\ttext\n", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes"):
        load_manifest(manifest, image_root)


def test_manifest_rejects_missing_images_and_malformed_rows(tmp_path):
    manifest = tmp_path / "labels.tsv"
    manifest.write_text("missing.png\ttext\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="image not found"):
        load_manifest(manifest, tmp_path)

    manifest.write_text("missing-column\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected image_path"):
        load_manifest(manifest, tmp_path)
