from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.space import CASpace


class CALoss(ABC):
    """Base class for loss/fitness functions."""

    @abstractmethod
    def __call__(self, output: CASpace, target: CASpace) -> float:
        pass
