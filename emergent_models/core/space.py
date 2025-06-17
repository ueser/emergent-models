import numpy as np
from typing import Tuple, Optional


# Core Data Structures (like torch.Tensor)
class CASpace:
    """Base class for cellular automata spaces (like torch.Tensor)"""
    def __init__(self, data: np.ndarray, device: str = 'cpu'):
        self.data = data.astype(np.uint8)  # Ensure uint8 for efficiency
        self.device = device
        self.shape = data.shape

    def to(self, device: str) -> 'CASpace':
        """Move space to device (CPU/GPU)"""
        # Implementation would handle device transfer
        return CASpace(self.data, device)

    def clone(self) -> 'CASpace':
        """Create a copy of the space"""
        return CASpace(self.data.copy(), self.device)

    def randomize(self, n_states: int = 2, rng: Optional[np.random.Generator] = None) -> None:
        """Fill space with random states"""
        if rng is None:
            rng = np.random.default_rng()
        self.data[:] = rng.integers(0, n_states, size=self.shape, dtype=np.uint8)

    def zeros(self) -> None:
        """Fill space with zeros"""
        self.data.fill(0)

    def ones(self) -> None:
        """Fill space with ones"""
        self.data.fill(1)

    def __eq__(self, other: 'CASpace') -> bool:
        """Check equality with another space"""
        return np.array_equal(self.data, other.data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape}, device='{self.device}')"


class Space1D(CASpace):
    """1D tape space for cellular automata"""
    def __init__(self, size: int, n_states: int = 2, device: str = 'cpu'):
        data = np.zeros(size, dtype=np.uint8)
        super().__init__(data, device)
        self.n_states = n_states
        self.size = size

    def get_neighborhood(self, index: int, radius: int = 1) -> np.ndarray:
        """Get neighborhood around a cell with boundary handling"""
        left = max(0, index - radius)
        right = min(self.size, index + radius + 1)

        # Handle boundaries by padding with zeros
        neighborhood = np.zeros(2 * radius + 1, dtype=np.uint8)
        data_slice = self.data[left:right]

        # Calculate offset for placing data in neighborhood
        offset = radius - (index - left)
        neighborhood[offset:offset + len(data_slice)] = data_slice

        return neighborhood

    def get_all_neighborhoods(self, radius: int = 1) -> np.ndarray:
        """Get all neighborhoods for efficient batch processing"""
        neighborhoods = np.zeros((self.size, 2 * radius + 1), dtype=np.uint8)
        for i in range(self.size):
            neighborhoods[i] = self.get_neighborhood(i, radius)
        return neighborhoods

    def set_pattern(self, pattern: np.ndarray, start_pos: int = 0) -> None:
        """Set a pattern at a specific position"""
        end_pos = min(start_pos + len(pattern), self.size)
        self.data[start_pos:end_pos] = pattern[:end_pos - start_pos]

    def randomize(self, n_states: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> None:
        """Fill space with random states"""
        if n_states is None:
            n_states = self.n_states
        super().randomize(n_states, rng)


class Space2D(CASpace):
    """2D grid space for cellular automata"""
    def __init__(self, height: int, width: int, n_states: int = 2, device: str = 'cpu'):
        data = np.zeros((height, width), dtype=np.uint8)
        super().__init__(data, device)
        self.n_states = n_states
        self.height = height
        self.width = width

    def get_neighborhood(self, row: int, col: int, radius: int = 1) -> np.ndarray:
        """Get Moore neighborhood around a cell"""
        size = 2 * radius + 1
        neighborhood = np.zeros((size, size), dtype=np.uint8)

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    neighborhood[dr + radius, dc + radius] = self.data[nr, nc]

        return neighborhood

    def randomize(self, n_states: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> None:
        """Fill space with random states"""
        if n_states is None:
            n_states = self.n_states
        super().randomize(n_states, rng)