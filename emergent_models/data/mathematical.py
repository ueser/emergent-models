"""
Mathematical Operation Datasets
===============================
Datasets for training cellular automata to perform mathematical operations
like doubling, increment, addition, etc.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

from .dataset import CADataset
from ..core.space import CASpace, Space1D
from ..encoders.position import PositionEncoder


class DoublingDataset(CADataset):
    """Dataset for training CA to double input numbers (x -> 2x)"""
    
    def __init__(self, input_range: Tuple[int, int] = (1, 30), window_size: int = 200):
        self.input_range = input_range
        self.window_size = window_size
        self.inputs = list(range(input_range[0], input_range[1] + 1))
        self.targets = [2 * x for x in self.inputs]
    
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[CASpace, CASpace]:
        input_val = self.inputs[idx]
        target_val = self.targets[idx]
        
        # Create input space (just the number for now - actual encoding happens in simulator)
        input_space = Space1D(1, n_states=64)
        input_space.data[0] = input_val
        
        # Create target space
        target_space = Space1D(1, n_states=64)
        target_space.data[0] = target_val
        
        return input_space, target_space


class IncrementDataset(CADataset):
    """Dataset for training CA to increment input numbers (x -> x+1)"""
    
    def __init__(self, input_range: Tuple[int, int] = (1, 32)):
        self.input_range = input_range
        self.inputs = list(range(input_range[0], input_range[1] + 1))
        self.targets = [x + 1 for x in self.inputs]
    
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[CASpace, CASpace]:
        input_val = self.inputs[idx]
        target_val = self.targets[idx]
        
        input_space = Space1D(1, n_states=64)
        input_space.data[0] = input_val
        
        target_space = Space1D(1, n_states=64)
        target_space.data[0] = target_val
        
        return input_space, target_space


class AdditionDataset(CADataset):
    """Dataset for training CA to add two numbers (x, y -> x+y)"""
    
    def __init__(self, input_range: Tuple[int, int] = (1, 15)):
        self.input_range = input_range
        self.data_pairs = []
        
        # Generate all pairs within range
        for x in range(input_range[0], input_range[1] + 1):
            for y in range(input_range[0], input_range[1] + 1):
                if x + y <= 30:  # Keep results reasonable
                    self.data_pairs.append((x, y, x + y))
    
    def __len__(self) -> int:
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[CASpace, CASpace]:
        x, y, target = self.data_pairs[idx]
        
        # Input space contains both numbers
        input_space = Space1D(2, n_states=64)
        input_space.data[0] = x
        input_space.data[1] = y
        
        # Target space contains the sum
        target_space = Space1D(1, n_states=64)
        target_space.data[0] = target
        
        return input_space, target_space


class MultiplicationDataset(CADataset):
    """Dataset for training CA to multiply two numbers (x, y -> x*y)"""
    
    def __init__(self, input_range: Tuple[int, int] = (1, 10)):
        self.input_range = input_range
        self.data_pairs = []
        
        # Generate all pairs within range
        for x in range(input_range[0], input_range[1] + 1):
            for y in range(input_range[0], input_range[1] + 1):
                if x * y <= 50:  # Keep results reasonable
                    self.data_pairs.append((x, y, x * y))
    
    def __len__(self) -> int:
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[CASpace, CASpace]:
        x, y, target = self.data_pairs[idx]
        
        input_space = Space1D(2, n_states=64)
        input_space.data[0] = x
        input_space.data[1] = y
        
        target_space = Space1D(1, n_states=64)
        target_space.data[0] = target
        
        return input_space, target_space


class SequenceDataset(CADataset):
    """Dataset for sequence-to-sequence tasks"""
    
    def __init__(self, sequences: List[Tuple[List[int], List[int]]], max_length: int = 20):
        self.sequences = sequences
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[CASpace, CASpace]:
        input_seq, target_seq = self.sequences[idx]
        
        # Pad sequences to max_length
        input_padded = input_seq + [0] * (self.max_length - len(input_seq))
        target_padded = target_seq + [0] * (self.max_length - len(target_seq))
        
        input_space = Space1D(self.max_length, n_states=64)
        input_space.data[:] = input_padded[:self.max_length]
        
        target_space = Space1D(self.max_length, n_states=64)
        target_space.data[:] = target_padded[:self.max_length]
        
        return input_space, target_space


class BinaryAdditionDataset(CADataset):
    """Dataset for binary addition tasks"""
    
    def __init__(self, bit_length: int = 8, n_samples: int = 1000):
        self.bit_length = bit_length
        self.n_samples = n_samples
        self.rng = np.random.default_rng()
        
        # Pre-generate samples
        self.samples = []
        for _ in range(n_samples):
            a = self.rng.integers(0, 2**bit_length)
            b = self.rng.integers(0, 2**bit_length)
            c = a + b
            
            # Convert to binary representations
            a_bin = [(a >> i) & 1 for i in range(bit_length)]
            b_bin = [(b >> i) & 1 for i in range(bit_length)]
            c_bin = [(c >> i) & 1 for i in range(bit_length + 1)]  # +1 for carry
            
            self.samples.append((a_bin + b_bin, c_bin))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[CASpace, CASpace]:
        input_bits, target_bits = self.samples[idx]
        
        input_space = Space1D(len(input_bits), n_states=2)
        input_space.data[:] = input_bits
        
        target_space = Space1D(len(target_bits), n_states=2)
        target_space.data[:] = target_bits
        
        return input_space, target_space


class PatternRecognitionDataset(CADataset):
    """Dataset for pattern recognition tasks"""
    
    def __init__(self, patterns: List[np.ndarray], labels: List[int], noise_level: float = 0.0):
        self.patterns = patterns
        self.labels = labels
        self.noise_level = noise_level
        self.rng = np.random.default_rng()
    
    def __len__(self) -> int:
        return len(self.patterns)
    
    def __getitem__(self, idx: int) -> Tuple[CASpace, CASpace]:
        pattern = self.patterns[idx].copy()
        label = self.labels[idx]
        
        # Add noise if specified
        if self.noise_level > 0:
            noise_mask = self.rng.random(pattern.shape) < self.noise_level
            pattern[noise_mask] = self.rng.integers(0, 2, size=noise_mask.sum())
        
        input_space = Space1D(len(pattern), n_states=2)
        input_space.data[:] = pattern
        
        # One-hot encode label
        n_classes = max(self.labels) + 1
        target_space = Space1D(n_classes, n_states=2)
        target_space.data[label] = 1
        
        return input_space, target_space


def create_fibonacci_dataset(length: int = 20) -> SequenceDataset:
    """Create a dataset for learning Fibonacci sequence"""
    sequences = []
    
    # Generate Fibonacci sequence
    fib = [1, 1]
    for i in range(2, length):
        fib.append(fib[i-1] + fib[i-2])
    
    # Create input-output pairs (predict next number)
    for i in range(len(fib) - 1):
        input_seq = fib[:i+1]
        target_seq = [fib[i+1]]
        sequences.append((input_seq, target_seq))
    
    return SequenceDataset(sequences)


def create_prime_dataset(max_num: int = 100) -> PatternRecognitionDataset:
    """Create a dataset for prime number recognition"""
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    patterns = []
    labels = []
    
    for num in range(2, max_num + 1):
        # Binary representation of number
        binary = np.array([(num >> i) & 1 for i in range(8)])  # 8-bit representation
        patterns.append(binary)
        labels.append(1 if is_prime(num) else 0)
    
    return PatternRecognitionDataset(patterns, labels)
