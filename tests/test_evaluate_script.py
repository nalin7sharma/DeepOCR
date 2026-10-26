import pytest

from scripts.evaluate import require_checkpoint


def test_evaluation_refuses_to_score_missing_untrained_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="random weights"):
        require_checkpoint(tmp_path / "missing.pth")


def test_evaluation_accepts_existing_checkpoint(tmp_path):
    checkpoint = tmp_path / "trained.pth"
    checkpoint.write_bytes(b"trained checkpoint placeholder")
    assert require_checkpoint(checkpoint) == checkpoint
