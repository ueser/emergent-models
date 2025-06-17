from __future__ import annotations

from typing import Dict, Tuple

from ..core.module import CAModule
from ..core.space import CASpace


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

    def forward(self, space: CASpace) -> CASpace:  # pragma: no cover - placeholder
        """Apply rules to a space."""
        # Actual implementation would apply rule_table to space
        return space
