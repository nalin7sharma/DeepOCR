"""Tests for multi-model word voting (ROVER)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from postprocessing.ensemble import Hypothesis, rover


def test_agreement_beats_a_single_stronger_model():
    text, words = rover([
        Hypothesis("the quick brown fax", "strong", 2.0),
        Hypothesis("the quick brown fox", "a", 1.5),
        Hypothesis("the quick brown fox", "b", 1.5),
    ])
    assert text == "the quick brown fox"
    assert words[-1].sources == ["a", "b"]
    assert words[-1].alternatives == {"fax": 0.4}


def test_strongest_model_wins_when_others_disagree_with_each_other():
    text, _ = rover([
        Hypothesis("assurances lively song", "strong", 3.0),
        Hypothesis("assuredhes lively song", "a", 1.0),
        Hypothesis("assurechess lively song", "b", 1.0),
    ])
    assert text == "assurances lively song"


def test_alignment_survives_missing_and_extra_words():
    text, words = rover([
        Hypothesis("he is also a director of garages", "a", 2.0),
        Hypothesis("he is a director of garages", "b", 1.5),       # missing "also"
        Hypothesis("he is also a director of of garages", "c", 1.5),  # extra "of"
    ])
    assert text == "he is also a director of garages"
    assert all(0 < w.support <= 1 for w in words)


def test_full_agreement_has_full_support():
    _, words = rover([Hypothesis("same text", "a", 1), Hypothesis("same text", "b", 2)])
    assert [w.support for w in words] == [1.0, 1.0]


def test_empty_input():
    assert rover([]) == ("", [])
    assert rover([Hypothesis("ignored", "a", 0)]) == ("", [])
