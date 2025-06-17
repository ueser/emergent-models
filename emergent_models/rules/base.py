from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from ..core.module import CAModule
from ..core.space import CASpace, Space1D


class RuleSet(CAModule):
    """Defines local cell update rules."""

    def __init__(self, neighborhood_size: int, n_states: int) -> None:
        super().__init__()
        self.neighborhood_size = neighborhood_size
        self.n_states = n_states
        self.rule_table: Dict[Tuple[int, ...], int] = {}

    def set_rule(self, pattern: Tuple[int, ...], new_state: int) -> None:
        """Define a mapping from a neighborhood pattern to a new state."""
        self.rule_table[pattern] = new_state

    def get_rule(self, pattern: Tuple[int, ...]) -> int:
        """Get the output state for a given pattern."""
        return self.rule_table.get(pattern, 0)  # Default to state 0

    def forward(self, space: CASpace) -> CASpace:
        """Apply rules to a space."""
        if isinstance(space, Space1D):
            return self._apply_1d(space)
        else:
            # For now, just return the space unchanged for other types
            return space.clone()

    def _apply_1d(self, space: Space1D) -> Space1D:
        """Apply 1D CA rules"""
        radius = self.neighborhood_size // 2
        new_space = Space1D(space.size, space.n_states, space.device)

        for i in range(space.size):
            neighborhood = space.get_neighborhood(i, radius)
            pattern = tuple(neighborhood)
            new_state = self.get_rule(pattern)
            new_space.data[i] = new_state

        return new_space
