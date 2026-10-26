import cv2
import numpy as np

from api.ocr_engine import TextRegion
from api.structure import reconstruct_page


def region(text, x1, y1, x2, y2, confidence=0.95):
    return TextRegion(
        polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        text=text,
        confidence=confidence,
    )


def test_groups_multiple_lines_in_reading_order():
    image = np.full((120, 300, 3), 255, dtype=np.uint8)
    result = reconstruct_page(
        image,
        [region("second line", 10, 50, 120, 70), region("first line", 10, 15, 100, 35)],
    )
    assert [block.kind for block in result.blocks] == ["text", "text"]
    assert result.markdown == "first line\nsecond line"


def test_reconstructs_markdown_table_from_separated_repeated_cells():
    image = np.full((100, 300, 3), 255, dtype=np.uint8)
    result = reconstruct_page(
        image,
        [
            region("Item", 10, 10, 55, 28),
            region("Amount", 140, 10, 205, 28),
            region("Paper", 10, 40, 65, 58),
            region("$4", 140, 40, 165, 58),
        ],
    )
    assert len(result.blocks) == 1
    block = result.blocks[0]
    assert block.kind == "table"
    assert block.cells == [["Item", "Amount"], ["Paper", "$4"]]
    assert block.markdown == "| Item | Amount |\n| --- | --- |\n| Paper | $4 |"
    assert block.review_required


def test_does_not_turn_separately_detected_words_into_a_table():
    image = np.full((80, 300, 3), 255, dtype=np.uint8)
    result = reconstruct_page(
        image,
        [
            region("Please", 10, 12, 47, 30),
            region("review", 53, 12, 97, 30),
            region("the", 102, 12, 125, 30),
            region("attached", 130, 12, 192, 30),
            region("form.", 198, 12, 235, 30),
        ],
    )
    assert [block.kind for block in result.blocks] == ["text"]
    assert result.markdown == "Please review the attached form."


def test_reconstructs_ruled_table_and_preserves_blank_cells():
    image = np.full((190, 340, 3), 255, dtype=np.uint8)
    for x in (30, 140, 240):
        cv2.line(image, (x, 25), (x, 145), (0, 0, 0), 2)
    for y in (25, 85, 145):
        cv2.line(image, (30, y), (240, y), (0, 0, 0), 2)

    result = reconstruct_page(
        image,
        [
            region("Name", 45, 43, 100, 65),
            region("Status", 155, 43, 215, 65),
            region("Ada", 45, 103, 90, 125),
        ],
    )
    tables = [block for block in result.blocks if block.kind == "table"]
    assert len(tables) == 1
    assert tables[0].cells == [["Name", "Status"], ["Ada", ""]]
    assert "| Ada |  |" in result.markdown
    assert tables[0].review_required


def test_adds_labeled_signature_placeholder_without_claiming_verification():
    image = np.full((80, 320, 3), 255, dtype=np.uint8)
    result = reconstruct_page(image, [region("Authorized signature:", 10, 20, 190, 40)])
    assert result.markdown == "Authorized signature: [Signature]"
    assert result.blocks[0].kind == "signature"
    assert result.blocks[0].review_required


def test_marks_low_confidence_text_for_review_but_preserves_source():
    image = np.full((80, 320, 3), 255, dtype=np.uint8)
    result = reconstruct_page(image, [region("uncertain words", 10, 20, 190, 40, 0.2)])
    block = result.blocks[0]
    assert result.markdown == "[Illegible]"
    assert block.kind == "illegible"
    assert block.raw_text == "uncertain words"
    assert block.review_required


def test_detects_unchecked_and_checked_boxes_next_to_labels():
    image = np.full((140, 360, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (40, 40), (0, 0, 0), 2)
    cv2.rectangle(image, (20, 75), (40, 95), (0, 0, 0), 2)
    cv2.line(image, (24, 84), (29, 90), (0, 0, 0), 2)
    cv2.line(image, (29, 90), (37, 79), (0, 0, 0), 2)

    result = reconstruct_page(
        image,
        [
            region("Email updates", 52, 20, 180, 40),
            region("Terms accepted", 52, 75, 175, 95),
        ],
    )
    assert [block.kind for block in result.blocks] == ["checkbox", "checkbox"]
    assert result.markdown == "[ ] Email updates\n[x] Terms accepted"
    assert [block.checkbox_state for block in result.blocks] == ["unchecked", "checked"]
    assert all(block.review_required for block in result.blocks)
    assert all(block.confidence == 0.7 for block in result.blocks)


def test_does_not_claim_confidence_for_ambiguous_visual_content():
    image = np.full((80, 200, 3), 255, dtype=np.uint8)
    result = reconstruct_page(image, [region("ok", 10, 20, 50, 40, 0.8)], review_threshold=0.81)
    assert result.blocks[0].review_required
    assert result.markdown == "[Illegible]"


def test_does_not_treat_square_letters_inside_text_as_checkboxes():
    image = np.full((100, 400, 3), 255, dtype=np.uint8)
    cv2.putText(image, "BUSINESS FORM", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    result = reconstruct_page(image, [region("BUSINESS FORM", 20, 30, 255, 64)])
    assert [block.kind for block in result.blocks] == ["text"]
    assert result.markdown == "BUSINESS FORM"


def test_tilted_lines_with_overlapping_boxes_stay_separate():
    # Two 6°-tilted lines 40 px apart: their axis-aligned boxes overlap heavily.
    def tilted(text, y):
        rise = 600 * 0.105
        return TextRegion(polygon=[[10, y + rise], [610, y], [610, y + 25], [10, y + rise + 25]],
                          text=text, confidence=0.95)

    image = np.full((260, 640, 3), 255, dtype=np.uint8)
    result = reconstruct_page(image, [tilted("second line", 140), tilted("first line", 100)])
    assert result.markdown == "first line\nsecond line"
