"""
Document Layout Analysis for DeepOCR.

Classifies detected text regions into structural roles (title, heading,
paragraph, table, column) and assigns a reading order.

The analysis is rule-based and uses spatial heuristics:
  - Region height relative to median  ->  title vs heading vs body
  - Horizontal overlap clustering     ->  multi-column detection
  - Density of regions                ->  table detection heuristic
  - Top-to-bottom, left-to-right sort ->  reading order

This module is independent of the OCR recognition step and operates on
bounding boxes produced by any text detector.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class LayoutBlock:
    """Represents a single detected region with its layout classification.

    Attributes:
        bbox: (x, y, w, h) bounding box.
        region_type: Semantic type: title | heading | paragraph | table_cell | unknown.
        reading_order: Integer position in the inferred reading sequence.
        column: Column index (0-based) if multi-column layout detected.
        score: Optional detector confidence.
        text: OCR result (filled later by the pipeline).
    """
    bbox: tuple[int, int, int, int]
    region_type: str = "paragraph"
    reading_order: int = 0
    column: int = 0
    score: float = -1.0
    text: str = ""


class LayoutAnalyzer:
    """Analyse spatial layout of detected text regions.

    Args:
        config: Optional dict with analysis parameters.

    Config keys (defaults):
        title_height_ratio   (2.0)  region height / median_height -> title if >=
        heading_height_ratio (1.5)  region height / median_height -> heading if >=
        column_gap_fraction  (0.05) min horizontal gap (as fraction of image width)
                                    to consider separate columns
        table_density        (0.6)  fraction of bounding grid cells that must be
                                    occupied to classify as table row
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.title_ratio = cfg.get("title_height_ratio", 2.0)
        self.heading_ratio = cfg.get("heading_height_ratio", 1.5)
        self.col_gap = cfg.get("column_gap_fraction", 0.05)

    def analyze(
        self,
        detections: list[dict],
        image_width: int = 0,
        image_height: int = 0,
    ) -> list[LayoutBlock]:
        """Analyse detected regions and return classified LayoutBlock list.

        Args:
            detections: List of dicts from text detector (each has "bbox", "score", "type").
            image_width: Source image width (used for column detection).
            image_height: Source image height.

        Returns:
            List of LayoutBlock, sorted by reading order.
        """
        if not detections:
            return []

        blocks = [
            LayoutBlock(
                bbox=d["bbox"],
                score=d.get("score", -1.0),
            )
            for d in detections
        ]

        # ── Compute median region height for relative sizing ───────────────
        heights = [b.bbox[3] for b in blocks]
        median_h = statistics.median(heights) if heights else 1

        # ── Classify each block ────────────────────────────────────────────
        for block in blocks:
            h = block.bbox[3]
            ratio = h / max(median_h, 1)
            if ratio >= self.title_ratio:
                block.region_type = "title"
            elif ratio >= self.heading_ratio:
                block.region_type = "heading"
            else:
                block.region_type = "paragraph"

        # ── Detect multi-column layout ─────────────────────────────────────
        if image_width > 0:
            blocks = self._assign_columns(blocks, image_width)

        # ── Assign reading order ───────────────────────────────────────────
        # Sort: column first, then top-to-bottom within each column.
        blocks.sort(key=lambda b: (b.column, b.bbox[1], b.bbox[0]))
        for i, block in enumerate(blocks):
            block.reading_order = i

        return blocks

    def _assign_columns(self, blocks: list[LayoutBlock], image_width: int) -> list[LayoutBlock]:
        """Group blocks into columns separated by empty vertical gutters.

        Approach:
          - Project every block onto the x-axis and find uncovered x-ranges
            (gutters) wider than ``col_gap * image_width`` between the text.
          - A gutter only counts if at least two blocks lie on each side
            (ragged single-column lines must not create fake columns).
          - Blocks are assigned to the column containing their centre;
            blocks that span a gutter (e.g. a full-width title) stay in
            column 0 and are therefore read first.
        """
        gap_threshold = max(1.0, self.col_gap * image_width)
        # Full-width blocks (titles, banners) cannot reveal columns, so ignore them here.
        narrow = [b for b in blocks if b.bbox[2] < 0.6 * image_width] or blocks
        intervals = sorted((b.bbox[0], b.bbox[0] + b.bbox[2]) for b in narrow)
        gutters: list[tuple[int, int]] = []
        reach = intervals[0][1] if intervals else 0
        for start, end in intervals[1:]:
            if start - reach > gap_threshold:
                left = sum(1 for b in narrow if b.bbox[0] + b.bbox[2] <= reach)
                right = sum(1 for b in narrow if b.bbox[0] >= start)
                if left >= 2 and right >= 2:
                    gutters.append((reach, start))
            reach = max(reach, end)

        for block in blocks:
            x1, x2 = block.bbox[0], block.bbox[0] + block.bbox[2]
            spans = any(x1 < g_start and x2 > g_end for g_start, g_end in gutters)
            cx = (x1 + x2) / 2
            block.column = 0 if spans else sum(1 for g_start, _ in gutters if cx > g_start)

        return blocks

    def to_text(self, blocks: list[LayoutBlock], separator: str = "\n") -> str:
        """Concatenate recognised text from blocks in reading order.

        Args:
            blocks: LayoutBlock list (already sorted by reading_order).
            separator: String inserted between blocks.

        Returns:
            Combined extracted text string.
        """
        parts = []
        for block in sorted(blocks, key=lambda b: b.reading_order):
            if block.text:
                parts.append(block.text)
        return separator.join(parts)
