from __future__ import annotations

from typing import List, Union

import numpy as np

from .base import CATransform
from ..core.space import CASpace, Space1D


class BinaryEncoder(CATransform):
    """Encode data to binary states."""

    def __call__(self, data: Union[str, List[int]]) -> CASpace:
        if isinstance(data, str):
            bits = [int(b) for b in data]
        else:
            bits = [int(x) for x in data]
        space = Space1D(size=len(bits), n_states=2)
        space.data[:] = np.array(bits, dtype=np.int32)
        return space
