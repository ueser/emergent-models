from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.space import CASpace


class CATransform(ABC):
    """Base class for data transformations."""

    @abstractmethod
    def __call__(self, data: Any) -> CASpace:
        pass
