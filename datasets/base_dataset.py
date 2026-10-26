"""Abstract base dataset for DeepOCR."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset


class BaseOCRDataset(Dataset, ABC):
    """Abstract dataset class all OCR datasets inherit from."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]: ...

    @property
    @abstractmethod
    def vocabulary(self) -> list[str]: ...
