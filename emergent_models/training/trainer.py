from __future__ import annotations

from typing import List

import numpy as np

from ..core.genome import Genome
from ..losses.base import CALoss
from ..optimizers.genetic import GAOptimizer
from ..simulators.base import Simulator
from ..data.dataloader import CADataLoader


class CATrainer:
    """High-level training interface."""

    def __init__(self, simulator: Simulator, optimizer: GAOptimizer, loss_fn: CALoss, device: str = "cpu") -> None:
        self.simulator = simulator
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def fit(self, population: List[Genome], dataloader: CADataLoader, epochs: int = 100) -> None:
        for epoch in range(epochs):
            fitness_scores: List[float] = []
            for batch in dataloader:
                batch_fitness = []
                for genome in population:
                    genome_fitness = 0.0
                    for input_state, target_state in batch:
                        output = self.simulator(genome, input_state)
                        loss = self.loss_fn(output, target_state)
                        genome_fitness += loss
                    batch_fitness.append(genome_fitness / len(batch))
                fitness_scores = batch_fitness
            self.optimizer.step(fitness_scores)
            print(f"Epoch {epoch}: Avg Fitness = {np.mean(fitness_scores):.4f}")
