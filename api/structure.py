"""Conservative layout reconstruction for PaddleOCR text and page images."""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from api.ocr_engine import TextRegion


SIGNATURE_LABEL = re.compile(
    r"\b(signature|signed\s+by|sign\s+here|authorized\s+signatory)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExtractionBlock:
    kind: str
    text: str
    markdown: str
    polygon: list[list[float]]
    confidence: float
    review_required: bool
    raw_text: str | None = None
    cells: list[list[str]] | None = None
    checkbox_state: str | None = None
    words: list[dict] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PageExtraction:
    text: str
    markdown: str
    blocks: list[ExtractionBlock]


@dataclass
class _Line:
    regions: list[TextRegion]
    bbox: tuple[float, float, float, float]

    @property
    def text(self) -> str:
        return " ".join(region.text.strip() for region in self.regions if region.text.strip())

    @property
    def confidence(self) -> float:
        if not self.regions:
            return 0.0
        return sum(region.confidence for region in self.regions) / len(self.regions)


@dataclass(frozen=True)
class _Grid:
    bbox: tuple[int, int, int, int]
    rows: list[int]
    columns: list[int]


def reconstruct_page(
    image: np.ndarray,
    regions: list[TextRegion],
    review_threshold: float = 0.45,
) -> PageExtraction:
    """Create reading-order blocks and faithful Markdown placeholders.

    Tables are reconstructed only when OCR itself exposes repeated, visibly
    separated cells in consecutive rows. Signature placeholders require a
    printed label; this is not signature or signer verification.
    """
    if not 0 <= review_threshold <= 1:
        raise ValueError("The OCR review threshold must be between 0 and 1.")

    lines = _group_lines(regions)
    # Typical body-text height, used to recognise titles and headings by size.
    line_heights = [max(1.0, line.bbox[3] - line.bbox[1]) for line in lines if line.text]
    glyph_widths = [_glyph_width(line) for line in lines if len(line.text) >= 6]
    body_height = (
        (float(np.median(line_heights)), float(np.median(glyph_widths)) if glyph_widths else None)
        if len(line_heights) >= 3 else None
    )
    checkboxes = _detect_checkboxes(image, lines)
    grid_tables = _grid_table_blocks(image, regions)
    grid_line_indices: set[int] = set()
    for block in grid_tables:
        x1, y1, x2, y2 = _region_bbox(
            TextRegion(block.polygon, "", block.confidence)
        )
        grid_line_indices.update(
            index
            for index, line in enumerate(lines)
            if line.bbox[0] >= x1 - 2
            and line.bbox[2] <= x2 + 2
            and line.bbox[1] >= y1 - 2
            and line.bbox[3] <= y2 + 2
        )
    consumed_checkboxes: set[int] = set()
    blocks: list[ExtractionBlock] = list(grid_tables)
    index = 0

    while index < len(lines):
        if index in grid_line_indices:
            index += 1
            continue
        table = _table_at(lines, index, review_threshold)
        if table is not None:
            block, next_index = table
            blocks.append(block)
            index = next_index
            continue

        line = lines[index]
        text = line.text
        if not text:
            index += 1
            continue

        polygon = _bbox_polygon(line.bbox)
        matched = _checkboxes_for_line(checkboxes, line, consumed_checkboxes)
        if matched:
            markdown_prefix = " ".join(item[1] for item in matched)
            consumed_checkboxes.update(item[0] for item in matched)
            markdown = f"{markdown_prefix} {text}".strip()
            kind = "checkbox"
            review = True
            state = (
                "uncertain"
                if any(item[2] == "uncertain" for item in matched)
                else ",".join(item[2] for item in matched)
            )
            confidence = min(line.confidence, *(item[3] for item in matched))
        elif SIGNATURE_LABEL.search(text):
            markdown = f"{text} [Signature]"
            kind = "signature"
            review = True
            state = None
            confidence = line.confidence
        elif line.confidence < review_threshold:
            markdown = "[Illegible]"
            kind = "illegible"
            review = True
            state = None
            confidence = line.confidence
        elif _heading_level(line, text, body_height):
            level = _heading_level(line, text, body_height)
            markdown = f"{'#' * level} {text}"
            kind = "title" if level == 1 else "heading"
            review = False
            state = None
            confidence = line.confidence
        else:
            markdown = text
            kind = "text"
            review = False
            state = None
            confidence = line.confidence

        blocks.append(
            ExtractionBlock(
                kind=kind,
                text=text if kind != "checkbox" else markdown,
                markdown=markdown,
                polygon=polygon,
                confidence=round(max(0.0, min(1.0, confidence)), 4),
                review_required=review,
                raw_text=text if kind == "illegible" else None,
                checkbox_state=state,
                words=_line_words(line),
            )
        )
        index += 1

    blocks.sort(key=lambda block: (min(point[1] for point in block.polygon), min(point[0] for point in block.polygon)))
    document_text = "\n".join(block.markdown for block in blocks)
    return PageExtraction(
        text=document_text,
        markdown=document_text,
        blocks=blocks,
    )


def _line_words(line: _Line) -> list[dict] | None:
    """Concatenated per-word consensus votes of a line's regions (ensemble engine only)."""
    if not any(region.words for region in line.regions):
        return None
    words: list[dict] = []
    for region in line.regions:
        words.extend(region.words or [{"text": w, "support": None, "sources": [], "alternatives": {}}
                                      for w in region.text.split()])
    return words


def _glyph_width(line: _Line) -> float:
    """Average character width — a font-size cue that tall handwriting does not inflate."""
    width = sum(_region_bbox(region)[2] - _region_bbox(region)[0] for region in line.regions)
    chars = sum(len(region.text.strip()) for region in line.regions)
    return width / max(1, chars)


def _heading_level(line: _Line, text: str, body: tuple[float, float | None] | None) -> int:
    """1 for titles, 2 for headings, 0 for body text (size-based layout rule)."""
    if body is None or len(text.split()) > 10 or len(text) < 3 or text.rstrip().endswith((".", ",", ";", ":")):
        return 0
    body_height, body_glyph = body
    height_ratio = (line.bbox[3] - line.bbox[1]) / max(1.0, body_height)
    glyph_ratio = _glyph_width(line) / body_glyph if body_glyph and len(text) >= 6 else 0.0
    size = max(height_ratio, glyph_ratio)
    if size >= 1.9:
        return 1
    if size >= 1.4:
        return 2
    return 0


def _region_bbox(region: TextRegion) -> tuple[float, float, float, float]:
    if not region.polygon:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [point[0] for point in region.polygon]
    ys = [point[1] for point in region.polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_polygon(bbox: tuple[float, float, float, float]) -> list[list[float]]:
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _geometry(region: TextRegion) -> tuple[float, float, float, float]:
    """Centre x/y, text height and baseline slope of a (possibly rotated) quad."""
    pts = region.polygon
    if len(pts) == 4:
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts
        height = (math.hypot(x3 - x0, y3 - y0) + math.hypot(x2 - x1, y2 - y1)) / 2
        dx = ((x1 - x0) + (x2 - x3)) / 2
        dy = ((y1 - y0) + (y2 - y3)) / 2
        slope = max(-0.5, min(0.5, dy / dx)) if abs(dx) > 1e-6 else 0.0
    else:
        bx1, by1, bx2, by2 = _region_bbox(region)
        height, slope = by2 - by1, 0.0
    cx = sum(point[0] for point in pts) / max(1, len(pts))
    cy = sum(point[1] for point in pts) / max(1, len(pts))
    return cx, cy, max(1.0, height), slope


def _group_lines(regions: list[TextRegion]) -> list[_Line]:
    """Group regions into text lines using their rotated geometry.

    Axis-aligned boxes of tilted text are much taller than the text itself, so
    neighbouring tilted lines would "overlap"; instead a region joins a line
    when its centre lies on that line's baseline (extrapolated with the slope).
    """
    geometry = {id(region): _geometry(region) for region in regions}
    ordered = sorted(regions, key=lambda region: (geometry[id(region)][1], geometry[id(region)][0]))
    lines: list[_Line] = []
    for region in ordered:
        x1, y1, x2, y2 = _region_bbox(region)
        cx, cy, height, _ = geometry[id(region)]
        match = None
        for line in reversed(lines):
            if y1 - line.bbox[3] > 3 * height:
                break
            anchor = min(line.regions, key=lambda member: abs(geometry[id(member)][0] - cx))
            acx, acy, aheight, aslope = geometry[id(anchor)]
            predicted = acy + aslope * (cx - acx)
            if abs(cy - predicted) <= 0.5 * min(height, aheight):
                match = line
                break
        if match is None:
            lines.append(_Line(regions=[region], bbox=(x1, y1, x2, y2)))
        else:
            match.regions.append(region)
            match.bbox = (
                min(match.bbox[0], x1),
                min(match.bbox[1], y1),
                max(match.bbox[2], x2),
                max(match.bbox[3], y2),
            )
    for line in lines:
        line.regions.sort(key=lambda region: _region_bbox(region)[0])
    return sorted(lines, key=lambda line: (sum(geometry[id(r)][1] for r in line.regions) / len(line.regions), line.bbox[0]))


def _cells_for_line(line: _Line) -> list[str] | None:
    if len(line.regions) < 2:
        return None
    sorted_regions = sorted(line.regions, key=lambda region: _region_bbox(region)[0])
    gaps = [
        _region_bbox(right)[0] - _region_bbox(left)[2]
        for left, right in zip(sorted_regions, sorted_regions[1:])
    ]
    line_height = max(1.0, line.bbox[3] - line.bbox[1])
    if not any(gap >= max(18.0, line_height * 1.25) for gap in gaps):
        return None
    cells = []
    for region in sorted_regions:
        cell_text = region.text.strip()
        if re.search(r"\s{2,}|\t|\s*[|¦]\s*", cell_text):
            cells.extend(part.strip() for part in re.split(r"\s{2,}|\t|\s*[|¦]\s*", cell_text))
        else:
            cells.append(cell_text)
    return cells if len(cells) >= 2 and all(cells) else None


def _table_at(
    lines: list[_Line],
    start: int,
    review_threshold: float,
) -> tuple[ExtractionBlock, int] | None:
    rows: list[tuple[_Line, list[str]]] = []
    index = start
    while index < len(lines):
        cells = _cells_for_line(lines[index])
        if cells is None or (rows and len(cells) != len(rows[0][1])):
            break
        rows.append((lines[index], cells))
        index += 1
    if len(rows) < 2:
        return None
    if len(rows[0][1]) == 2 and all(cells[0].rstrip().endswith(":") for _, cells in rows):
        return None  # "Label: value" form fields, not a table

    column_count = len(rows[0][1])
    lines_markdown = [
        "| " + " | ".join(cell.replace("|", r"\|") for cell in rows[0][1]) + " |",
        "| " + " | ".join("---" for _ in range(column_count)) + " |",
    ]
    lines_markdown.extend(
        "| " + " | ".join(cell.replace("|", r"\|") for cell in cells) + " |"
        for _, cells in rows[1:]
    )
    combined_bbox = (
        min(line.bbox[0] for line, _ in rows),
        min(line.bbox[1] for line, _ in rows),
        max(line.bbox[2] for line, _ in rows),
        max(line.bbox[3] for line, _ in rows),
    )
    confidence = min(line.confidence for line, _ in rows)
    table_rows = [cells for _, cells in rows]
    markdown = "\n".join(lines_markdown)
    return (
        ExtractionBlock(
            kind="table",
            text=markdown,
            markdown=markdown,
            polygon=_bbox_polygon(combined_bbox),
            confidence=round(max(0.0, min(1.0, confidence)), 4),
            review_required=True,
            cells=table_rows,
        ),
        index,
    )


def _grid_table_blocks(
    image: np.ndarray,
    regions: list[TextRegion],
) -> list[ExtractionBlock]:
    grids = _find_grid_regions(image)
    blocks = []
    for grid in grids:
        x1, y1, x2, y2 = grid.bbox
        row_count, column_count = len(grid.rows) - 1, len(grid.columns) - 1
        cells = [["" for _ in range(column_count)] for _ in range(row_count)]
        cell_confidences: list[float] = []
        for region in regions:
            rx1, ry1, rx2, ry2 = _region_bbox(region)
            center_x, center_y = (rx1 + rx2) / 2, (ry1 + ry2) / 2
            if not (x1 <= center_x <= x2 and y1 <= center_y <= y2):
                continue
            row = next(
                (idx for idx in range(row_count) if grid.rows[idx] <= center_y <= grid.rows[idx + 1]),
                None,
            )
            column = next(
                (idx for idx in range(column_count) if grid.columns[idx] <= center_x <= grid.columns[idx + 1]),
                None,
            )
            if row is None or column is None or not region.text.strip():
                continue
            existing = cells[row][column]
            cells[row][column] = f"{existing} {region.text.strip()}".strip()
            cell_confidences.append(region.confidence)

        markdown_lines = [
            "| " + " | ".join(_escape_cell(cell) for cell in cells[0]) + " |",
            "| " + " | ".join("---" for _ in range(column_count)) + " |",
        ]
        markdown_lines.extend(
            "| " + " | ".join(_escape_cell(cell) for cell in row) + " |"
            for row in cells[1:]
        )
        markdown = "\n".join(markdown_lines)
        blocks.append(
            ExtractionBlock(
                kind="table",
                text=markdown,
                markdown=markdown,
                polygon=_bbox_polygon((x1, y1, x2, y2)),
                confidence=round(min(cell_confidences), 4) if cell_confidences else 0.0,
                review_required=True,
                cells=cells,
            )
        )
    return blocks


def _escape_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ").strip()


def _find_grid_regions(image: np.ndarray) -> list[_Grid]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if gray.dtype != np.uint8:
        gray = cv2.convertScaleAbs(gray)
    image_height, image_width = gray.shape
    if image_width < 90 or image_height < 60:
        return []
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(20, image_width // 30), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(20, image_height // 30))
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    combined = cv2.bitwise_or(horizontal, vertical)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grids = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < 65 or height < 42:
            continue
        horizontal_crop = horizontal[y:y + height, x:x + width]
        vertical_crop = vertical[y:y + height, x:x + width]
        row_values = _line_coordinates(
            np.flatnonzero(horizontal_crop.sum(axis=1) >= width * 0.48) + y
        )
        column_values = _line_coordinates(
            np.flatnonzero(vertical_crop.sum(axis=0) >= height * 0.48) + x
        )
        if len(row_values) < 3 or len(column_values) < 3:
            continue
        if (
            any(right - left < 10 for left, right in zip(row_values, row_values[1:]))
            or any(right - left < 10 for left, right in zip(column_values, column_values[1:]))
        ):
            continue
        grid = _Grid((x, y, x + width, y + height), row_values, column_values)
        if not any(
            abs(existing.bbox[0] - x) < 5 and abs(existing.bbox[1] - y) < 5
            for existing in grids
        ):
            grids.append(grid)
    return grids


def _line_coordinates(values: np.ndarray) -> list[int]:
    if values.size == 0:
        return []
    groups = np.split(values, np.flatnonzero(np.diff(values) > 3) + 1)
    return [int(round(float(group.mean()))) for group in groups if group.size]


def _detect_checkboxes(
    image: np.ndarray,
    lines: list[_Line],
) -> list[tuple[int, int, int, int, str]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if gray.dtype != np.uint8:
        gray = cv2.convertScaleAbs(gray)
    if not lines:
        return []
    typical_height = float(np.median([max(1.0, line.bbox[3] - line.bbox[1]) for line in lines]))
    minimum_side = max(9, int(typical_height * 0.42))
    maximum_side = min(100, max(minimum_side + 1, int(typical_height * 2.2)))
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[int, int, int, int, str]] = []
    for contour_index, contour in enumerate(contours):
        if hierarchy is not None and hierarchy[0][contour_index][3] != -1:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, perimeter * 0.04, True)
        x, y, width, height = cv2.boundingRect(contour)
        if (
            len(approx) != 4
            or not cv2.isContourConvex(approx)
            or min(width, height) < minimum_side
            or max(width, height) > maximum_side
            or min(width, height) == 0
            or not 0.82 <= width / height <= 1.22
        ):
            continue
        inset_x, inset_y = max(2, width // 4), max(2, height // 4)
        interior = binary[y + inset_y:y + height - inset_y, x + inset_x:x + width - inset_x]
        if interior.size == 0:
            continue
        ink_ratio = float(np.count_nonzero(interior)) / interior.size
        state = "checked" if ink_ratio >= 0.045 else "unchecked" if ink_ratio <= 0.015 else "uncertain"
        if any(
            abs(x - old_x) < minimum_side and abs(y - old_y) < minimum_side
            for old_x, old_y, _, _, _ in candidates
        ):
            continue
        candidates.append((x, y, width, height, state))
    return candidates


def _checkboxes_for_line(
    checkboxes: list[tuple[int, int, int, int, str]],
    line: _Line,
    consumed: set[int],
) -> list[tuple[int, str, str, float]]:
    x1, y1, x2, y2 = line.bbox
    line_height = max(1.0, y2 - y1)
    matches = []
    for index, (x, y, width, height, state) in enumerate(checkboxes):
        if index in consumed:
            continue
        center_y = y + height / 2
        if abs(center_y - (y1 + y2) / 2) > max(height, line_height) * 0.8:
            continue
        if (
            x > x1 + line_height * 0.8
            or x + width < x1 - line_height * 4
        ):
            continue
        confidence = 0.5 if state == "uncertain" else 0.7
        marker = {"checked": "[x]", "unchecked": "[ ]", "uncertain": "[?]"}[state]
        matches.append((index, marker, state, confidence))
    return sorted(matches, key=lambda item: checkboxes[item[0]][0])
