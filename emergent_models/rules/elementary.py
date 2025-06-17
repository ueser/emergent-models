from __future__ import annotations

from .base import RuleSet


class ElementaryCA(RuleSet):
    """Elementary cellular automata (1D binary with 3-cell neighborhood)."""

    def __init__(self, rule_number: int) -> None:
        super().__init__(neighborhood_size=3, n_states=2)
        self.rule_number = rule_number
        self._initialize_rule_table()

    def _initialize_rule_table(self) -> None:
        for i in range(8):
            pattern = tuple(int(x) for x in format(i, "03b"))
            new_state = (self.rule_number >> i) & 1
            self.set_rule(pattern, new_state)
