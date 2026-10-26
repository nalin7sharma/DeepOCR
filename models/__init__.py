"""OCR model implementations for DeepOCR."""
def build_cnn_bilstm(num_classes, config=None):
    from .cnn_bilstm import build_cnn_bilstm as _b
    return _b(num_classes, config)

def build_crnn(num_classes, config=None):
    from .crnn import build_crnn as _b
    return _b(num_classes, config)

def build_transformer_ocr(num_classes, config=None):
    from .transformer_ocr import build_transformer_ocr as _b
    return _b(num_classes, config)

__all__ = ["build_cnn_bilstm", "build_crnn", "build_transformer_ocr"]
