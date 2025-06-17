from __future__ import annotations

from .base import CALoss
from ..core.space import CASpace


class PatternMatchLoss(CALoss):
    """Loss based on pattern matching."""

    def __call__(self, output: CASpace, target: CASpace) -> float:
        # Placeholder pattern matching implementation
        return float((output.data != target.data).sum())
