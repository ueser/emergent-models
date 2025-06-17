from __future__ import annotations

import numpy as np

from .base import CALoss
from ..core.space import CASpace


class HammingLoss(CALoss):
    """Hamming distance loss."""

    def __call__(self, output: CASpace, target: CASpace) -> float:
        return float(np.sum(output.data != target.data))
