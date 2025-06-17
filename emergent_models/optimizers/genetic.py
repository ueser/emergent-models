from __future__ import annotations

from typing import List

from ..core.genome import Genome


class GAOptimizer:
    """Genetic Algorithm optimizer."""

    def __init__(self, population_size: int = 100, mutation_rate: float = 0.01) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: List[Genome] = []

    def step(self, fitness_scores: List[float]) -> None:  # pragma: no cover - placeholder
        self._selection(fitness_scores)
        self._crossover()
        self._mutation()

    def _selection(self, fitness_scores: List[float]) -> None:  # pragma: no cover - placeholder
        pass

    def _crossover(self) -> None:  # pragma: no cover - placeholder
        pass

    def _mutation(self) -> None:  # pragma: no cover - placeholder
        pass
