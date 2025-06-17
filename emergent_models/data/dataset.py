from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from ..core.space import CASpace


class CADataset(ABC):
    """Base class for CA datasets."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[CASpace, CASpace]:
        """Return (input, target) pair."""
        pass
