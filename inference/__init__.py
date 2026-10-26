"""Inference pipeline for DeepOCR."""
def get_pipeline():
    from .ocr_pipeline import OCRPipeline
    return OCRPipeline

__all__ = ["get_pipeline"]
