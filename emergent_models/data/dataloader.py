from __future__ import annotations

from typing import Iterator, List

import numpy as np

from .dataset import CADataset


class CADataLoader:
    """DataLoader for CA datasets."""

    def __init__(self, dataset: CADataset, batch_size: int = 32, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[List[tuple[object, object]]]:
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch = [self.dataset[int(idx)] for idx in batch_indices]
            yield batch
