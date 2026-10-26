"""Tests for CTC decoding utilities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest
from training.ctc_utils import ctc_greedy_decode, ctc_beam_decode

BLANK = 0
IDX2CHAR = {0: "<BLANK>", 1: "a", 2: "b", 3: "c"}


def make_log_probs(T, B, C, forced_class=None):
    """Make log-probs with optionally forced argmax class."""
    probs = torch.zeros(T, B, C)
    if forced_class is not None:
        probs[:, :, forced_class] = 10.0
    import torch.nn.functional as F
    return F.log_softmax(probs, dim=2)


class TestGreedyDecode:
    def test_blank_only(self):
        lp = make_log_probs(5, 1, 4, forced_class=BLANK)
        result = ctc_greedy_decode(lp, IDX2CHAR)
        assert result == [""]

    def test_single_char(self):
        lp = make_log_probs(5, 1, 4, forced_class=1)  # all "a"
        result = ctc_greedy_decode(lp, IDX2CHAR)
        assert result == ["a"]  # repeated collapsed to one

    def test_batch_size(self):
        lp = make_log_probs(5, 3, 4, forced_class=BLANK)
        result = ctc_greedy_decode(lp, IDX2CHAR)
        assert len(result) == 3


class TestBeamDecode:
    def test_blank_only(self):
        lp = make_log_probs(5, 1, 4, forced_class=BLANK)
        result = ctc_beam_decode(lp, IDX2CHAR, beam_width=3)
        assert result == [""]

    def test_returns_list_of_strings(self):
        lp = make_log_probs(5, 2, 4)
        result = ctc_beam_decode(lp, IDX2CHAR, beam_width=3)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, str) for r in result)
