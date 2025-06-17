from __future__ import annotations

from typing import List

import numpy as np

from .base import CATransform
from ..core.space import CASpace, Space1D


class PositionEncoder(CATransform):
    """Encode data using position indexing."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __call__(self, data: List[int]) -> CASpace:
        space = Space1D(size=len(data), n_states=self.vocab_size)
        space.data[:] = np.array(data, dtype=np.int32)
        return space
