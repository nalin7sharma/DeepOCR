import pytest

from api.ocr_engine import OcrEngineError, _normalize_result


def test_normalizes_paddleocr_regions():
    result = [[
        [[[1, 2], [30, 2], [30, 12], [1, 12]], ("recognized", 0.91)],
    ]]
    regions = _normalize_result(result)
    assert len(regions) == 1
    assert regions[0].text == "recognized"
    assert regions[0].confidence == 0.91
    assert regions[0].polygon[0] == [1.0, 2.0]


def test_empty_page_is_a_valid_empty_result():
    assert _normalize_result([None]) == []


def test_rejects_malformed_result_coordinates():
    with pytest.raises(OcrEngineError, match="malformed"):
        _normalize_result([[[["bad"], ("text", 0.5)]]])


def test_rejects_invalid_normalized_confidence():
    with pytest.raises(OcrEngineError, match="non-finite"):
        _normalize_result([[[[[1, 2], [4, 2], [4, 5], [1, 5]], ("bad", float("nan"))]]])
