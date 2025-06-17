"""
EM-4/3 Cellular Automaton Rules
===============================
Implementation of the 4-state, radius-1 cellular automaton rules
as described in the EM-4/3 system.

States:
- 0: Empty/White
- 1: Active/Black  
- 2: Red (beacon)
- 3: Blue (separator/halt marker)
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from .base import RuleSet
from ..core.space import CASpace, Space1D


def lut_idx(l: int, c: int, r: int) -> int:
    """Convert 3-tuple (left, center, right) to lookup table index (0-63)"""
    return (l << 4) | (c << 2) | r


# Hard-wired immutable LUT entries for EM-4/3
_IMMUTABLE_RULES = {
    lut_idx(0, 0, 0): 0,  # 000 -> 0
    lut_idx(0, 2, 0): 2,  # 020 -> 2 (preserve red beacons)
    lut_idx(0, 0, 2): 0,  # 002 -> 0
    lut_idx(2, 0, 0): 0,  # 200 -> 0
    lut_idx(0, 3, 3): 3,  # 033 -> 3 (preserve blue separators)
    lut_idx(3, 3, 0): 3,  # 330 -> 3
    lut_idx(0, 0, 3): 0,  # 003 -> 0
    lut_idx(3, 0, 0): 0,  # 300 -> 0
}


def sanitize_rule(rule: np.ndarray) -> np.ndarray:
    """Sanitize rule array by enforcing immutable entries and clipping to 0-3"""
    rule = rule.astype(np.uint8, copy=True)
    
    # Enforce immutable rules
    for idx, value in _IMMUTABLE_RULES.items():
        rule[idx] = value
    
    # Clip values to valid range [0, 3]
    rule = np.clip(rule, 0, 3)
    
    return rule


def sanitize_programme(prog: np.ndarray) -> np.ndarray:
    """Remove accidental blue cells from programme (only 0, 1, 2 allowed)"""
    prog = prog.astype(np.uint8, copy=True)
    prog[prog == 3] = 0  # Convert blue to empty
    return prog


class EM43Rule(RuleSet):
    """EM-4/3 cellular automaton rule system"""
    
    def __init__(self, rule_array: np.ndarray = None) -> None:
        super().__init__(neighborhood_size=3, n_states=4)
        
        if rule_array is None:
            # Initialize with random rules
            rng = np.random.default_rng()
            rule_array = rng.integers(0, 4, 64, dtype=np.uint8)
        
        # Sanitize and store the rule array
        self.rule_array = sanitize_rule(rule_array)
        self._build_rule_table()
    
    def _build_rule_table(self) -> None:
        """Build the rule table from the rule array"""
        self.rule_table.clear()
        for i in range(64):
            # Convert index back to (left, center, right) pattern
            left = (i >> 4) & 3
            center = (i >> 2) & 3
            right = i & 3
            pattern = (left, center, right)
            self.rule_table[pattern] = self.rule_array[i]
    
    def set_rule_array(self, rule_array: np.ndarray) -> None:
        """Set a new rule array"""
        self.rule_array = sanitize_rule(rule_array)
        self._build_rule_table()
    
    def get_rule_array(self) -> np.ndarray:
        """Get the current rule array"""
        return self.rule_array.copy()
    
    def mutate(self, mutation_rate: float = 0.03, rng: np.random.Generator = None) -> 'EM43Rule':
        """Create a mutated copy of this rule"""
        if rng is None:
            rng = np.random.default_rng()
        
        new_rule = self.rule_array.copy()
        
        # Apply mutations
        mask = rng.random(64) < mutation_rate
        if mask.any():
            new_rule[mask] = rng.integers(0, 4, mask.sum(), dtype=np.uint8)
        
        return EM43Rule(new_rule)
    
    def crossover(self, other: 'EM43Rule', rng: np.random.Generator = None) -> 'EM43Rule':
        """Create offspring through crossover with another rule"""
        if rng is None:
            rng = np.random.default_rng()
        
        # Single-point crossover
        cut_point = rng.integers(1, 64)
        child_rule = np.concatenate([
            self.rule_array[:cut_point],
            other.rule_array[cut_point:]
        ])
        
        return EM43Rule(child_rule)
    
    def forward(self, space: CASpace) -> CASpace:
        """Apply EM-4/3 rules to the space"""
        if not isinstance(space, Space1D):
            raise ValueError("EM43Rule only supports 1D spaces")
        
        return self._apply_1d(space)
    
    def _apply_1d(self, space: Space1D) -> Space1D:
        """Apply 1D EM-4/3 rules efficiently"""
        new_space = Space1D(space.size, 4, space.device)
        
        # Handle boundary conditions (first and last cells stay 0)
        new_space.data[0] = 0
        new_space.data[-1] = 0
        
        # Apply rules to interior cells
        for i in range(1, space.size - 1):
            left = space.data[i - 1]
            center = space.data[i]
            right = space.data[i + 1]
            
            # Use direct array lookup for efficiency
            idx = lut_idx(left, center, right)
            new_space.data[i] = self.rule_array[idx]
        
        return new_space
    
    def __repr__(self) -> str:
        return f"EM43Rule(rule_array={self.rule_array})"


class EM43Genome:
    """Complete EM-4/3 genome: rule + programme"""
    
    def __init__(self, rule: EM43Rule, programme: np.ndarray = None) -> None:
        self.rule = rule
        
        if programme is None:
            # Default empty programme
            programme = np.zeros(10, dtype=np.uint8)
        
        self.programme = sanitize_programme(programme)
        self.fitness = 0.0
    
    def mutate_programme(self, mutation_rate: float = 0.08, rng: np.random.Generator = None) -> None:
        """Mutate the programme in-place"""
        if rng is None:
            rng = np.random.default_rng()
        
        mask = rng.random(len(self.programme)) < mutation_rate
        if mask.any():
            # Choose from states 0, 1, 2 (no blue in programme)
            self.programme[mask] = rng.choice([0, 1, 2], size=mask.sum(), p=[0.7, 0.2, 0.1])
    
    def clone(self) -> 'EM43Genome':
        """Create a deep copy of this genome"""
        return EM43Genome(
            EM43Rule(self.rule.get_rule_array()),
            self.programme.copy()
        )
    
    def __repr__(self) -> str:
        return f"EM43Genome(programme_length={len(self.programme)}, fitness={self.fitness:.3f})"
