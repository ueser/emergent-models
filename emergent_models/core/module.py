from abc import ABC, abstractmethod
from .space import CASpace

# Modules (like torch.nn.Module)
class CAModule(ABC):
    """Base class for all CA components (like nn.Module)"""
    def __init__(self):
        self._device = 'cpu'
    
    @abstractmethod
    def forward(self, x: CASpace) -> CASpace:
        pass
    
    def __call__(self, x: CASpace) -> CASpace:
        return self.forward(x)
    
    def to(self, device: str) -> 'CAModule':
        self._device = device
        return self
