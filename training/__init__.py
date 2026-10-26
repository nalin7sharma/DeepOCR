"""Training utilities for DeepOCR."""
from .ctc_utils import ctc_greedy_decode, ctc_beam_decode

def get_trainer():
    from .trainer import Trainer
    return Trainer

__all__ = ["ctc_greedy_decode", "ctc_beam_decode", "get_trainer"]
