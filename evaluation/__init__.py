"""Evaluation utilities for DeepOCR."""
from .metrics import cer_score, wer_score, recognition_accuracy, batch_cer, batch_wer

# Evaluator is imported lazily to avoid torch import at module level
def get_evaluator():
    from .evaluator import Evaluator
    return Evaluator

__all__ = ["cer_score", "wer_score", "recognition_accuracy",
           "batch_cer", "batch_wer", "get_evaluator"]
