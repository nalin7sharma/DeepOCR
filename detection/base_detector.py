"""Abstract base class for text detectors."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class BaseTextDetector(ABC):
    """Interface all text detectors must implement."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[dict]:
        """Detect text regions in image.

        Args:
            image: BGR numpy image.

        Returns:
            List of dicts with keys:
                bbox   : (x, y, w, h)
                score  : float confidence (0-1), -1 if unavailable
                type   : "text_line" | "text_block" | "word"
        """
        ...
