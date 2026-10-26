"""Unit tests for CER and WER metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from evaluation.metrics import cer_score, wer_score, recognition_accuracy


class TestCER:
    def test_perfect_match(self):
        assert cer_score("hello", "hello") == 0.0

    def test_complete_mismatch(self):
        # All characters wrong: distance = len(gt)
        assert cer_score("", "hello") == 1.0

    def test_one_substitution(self):
        cer = cer_score("helo", "hello")
        assert 0 < cer < 1.0

    def test_empty_gt(self):
        # Empty gt with empty pred -> 0
        assert cer_score("", "") == 0.0

    def test_empty_pred_nonempty_gt(self):
        assert cer_score("", "abc") == 1.0

    def test_single_insertion(self):
        # pred has one extra char
        cer = cer_score("helloo", "hello")
        assert cer == pytest.approx(1/5, abs=0.01)

    def test_single_deletion(self):
        cer = cer_score("helo", "hello")
        assert cer == pytest.approx(1/5, abs=0.01)


class TestWER:
    def test_perfect_match(self):
        assert wer_score("hello world", "hello world") == 0.0

    def test_one_wrong_word(self):
        wer = wer_score("hello wörld", "hello world")
        assert wer == pytest.approx(1/2, abs=0.01)

    def test_empty_both(self):
        assert wer_score("", "") == 0.0

    def test_empty_pred(self):
        assert wer_score("", "one two three") == 1.0

    def test_extra_word(self):
        wer = wer_score("hello world extra", "hello world")
        assert wer == pytest.approx(1/2, abs=0.01)


class TestRecognitionAccuracy:
    def test_all_correct(self):
        assert recognition_accuracy(["a", "b"], ["a", "b"]) == 1.0

    def test_none_correct(self):
        assert recognition_accuracy(["x", "y"], ["a", "b"]) == 0.0

    def test_half_correct(self):
        assert recognition_accuracy(["a", "x"], ["a", "b"]) == 0.5

    def test_empty(self):
        assert recognition_accuracy([], []) == 0.0
