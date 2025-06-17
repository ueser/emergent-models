from __future__ import annotations

from typing import Callable, Optional

from ..core.module import CAModule
from ..core.space import CASpace
from ..core.genome import Genome


class Simulator(CAModule):
    """Simulates CA evolution."""

    def __init__(self, max_steps: int = 100) -> None:
        super().__init__()
        self.max_steps = max_steps

    def forward(
        self,
        genome: Genome,
        initial_state: CASpace,
        halting_condition: Optional[Callable[[CASpace, int], bool]] = None,
    ) -> CASpace:
        state = initial_state.clone()
        for step in range(self.max_steps):
            state = genome(state)
            if halting_condition and halting_condition(state, step):
                break
        return state
