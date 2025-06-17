from __future__ import annotations

from typing import Optional

from .module import CAModule
from .space import CASpace
from ..rules.base import RuleSet


class Genome(CAModule):
    """Complete CA model: program + ruleset"""

    def __init__(self, ruleset: RuleSet, program: Optional[CASpace] = None) -> None:
        super().__init__()
        self.ruleset = ruleset
        self.program = program
        self.fitness = 0.0

    def forward(self, initial_state: CASpace) -> CASpace:
        """Run the CA with this genome"""
        if self.program is not None:
            state = self._combine_program_input(initial_state, self.program)
        else:
            state = initial_state
        return self.ruleset(state)

    def _combine_program_input(self, input_state: CASpace, program: CASpace) -> CASpace:
        """Combine program and input (implementation specific)"""
        # Placeholder implementation
        return input_state
