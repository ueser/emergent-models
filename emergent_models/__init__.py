"""
Emergent Models - A PyTorch-like library for cellular automata
"""

from .__version__ import __version__

# Core imports (like torch.tensor, torch.nn.Module)
from .core.space import CASpace, Space1D, Space2D
from .core.module import CAModule
from .core.genome import Genome

# Rules (like torch.nn layers)
from .rules.elementary import ElementaryCA
from .rules.base import RuleSet
from .rules.em43 import EM43Rule, EM43Genome

# Simulators (like torch inference)
from .simulators.base import Simulator
from .simulators.sequential import SequentialSimulator
from .simulators.numba_simulator import NumbaSimulator

# Training components (like torch.optim, torch.utils.data)
from .optimizers.genetic import GAOptimizer
from .data.dataset import CADataset
from .data.dataloader import CADataLoader
from .training.trainer import CATrainer, create_accuracy_validator

# Loss functions (like torch.nn.functional)
from .losses.distance import HammingLoss
from .losses.pattern import PatternMatchLoss

# Encoders (like torchvision.transforms)
from .encoders.binary import BinaryEncoder
from .encoders.position import PositionEncoder

# Utilities
from .utils.visualization import visualize_evolution
from .training.checkpointing import save_genome, load_genome

# Convenience imports
from . import rules
from . import optimizers
from . import losses
from . import encoders
from . import utils

__all__ = [
    # Core
    'CASpace', 'Space1D', 'Space2D', 'CAModule', 'Genome',

    # Rules
    'RuleSet', 'ElementaryCA', 'EM43Rule', 'EM43Genome',

    # Simulation
    'Simulator', 'SequentialSimulator', 'NumbaSimulator',

    # Training
    'GAOptimizer', 'CADataset', 'CADataLoader', 'CATrainer', 'create_accuracy_validator',

    # Losses
    'HammingLoss', 'PatternMatchLoss',

    # Encoders
    'BinaryEncoder', 'PositionEncoder',

    # Utils
    'visualize_evolution', 'save_genome', 'load_genome',

    # Modules
    'rules', 'optimizers', 'losses', 'encoders', 'utils',

    # Version
    '__version__',
]

