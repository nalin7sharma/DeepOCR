"""Dataset utilities for DeepOCR."""
from .augmentations import OCRAugmentor

def build_vocabulary(charset):
    from .ocr_dataset import build_vocabulary as _bv
    return _bv(charset)

def get_ocr_dataset():
    from .ocr_dataset import OCRDataset
    return OCRDataset

__all__ = ["OCRAugmentor", "build_vocabulary", "get_ocr_dataset"]
