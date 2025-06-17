import numpy as np


# Core Data Structures (like torch.Tensor)
class CASpace:
    """Base class for cellular automata spaces (like torch.Tensor)"""
    def __init__(self, data: np.ndarray, device: str = 'cpu'):
        self.data = data
        self.device = device
        self.shape = data.shape
        
    def to(self, device: str) -> 'CASpace':
        """Move space to device (CPU/GPU)"""
        # Implementation would handle device transfer
        return CASpace(self.data, device)
    
    def clone(self) -> 'CASpace':
        """Create a copy of the space"""
        return CASpace(self.data.copy(), self.device)


class Space1D(CASpace):
    """1D tape space"""
    def __init__(self, size: int, n_states: int = 2, device: str = 'cpu'):
        data = np.zeros(size, dtype=np.int32)
        super().__init__(data, device)
        self.n_states = n_states


class Space2D(CASpace):
    """2D grid space"""
    def __init__(self, height: int, width: int, n_states: int = 2, device: str = 'cpu'):
        data = np.zeros((height, width), dtype=np.int32)
        super().__init__(data, device)
        self.n_states = n_states