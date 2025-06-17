from __future__ import annotations

from typing import List, Union, Optional
import numpy as np
import math

from ..core.genome import Genome
from ..rules.em43 import EM43Genome, EM43Rule


class GAOptimizer:
    """Genetic Algorithm optimizer with tournament selection, elitism, and random immigrants."""

    def __init__(
        self,
        population_size: int = 100,
        mutation_rate: float = 0.01,
        programme_mutation_rate: float = 0.08,
        elite_fraction: float = 0.1,
        tournament_size: int = 3,
        random_immigrant_rate: float = 0.2,
        sparsity_penalty: float = 0.01
    ) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.programme_mutation_rate = programme_mutation_rate
        self.elite_fraction = elite_fraction
        self.tournament_size = tournament_size
        self.random_immigrant_rate = random_immigrant_rate
        self.sparsity_penalty = sparsity_penalty

        self.population: List[Union[Genome, EM43Genome]] = []
        self.fitness_scores: List[float] = []
        self.generation = 0
        self.rng = np.random.default_rng()

        # Calculate derived parameters
        self.n_elite = max(1, int(math.ceil(self.elite_fraction * self.population_size)))
        self.n_immigrants = max(1, int(self.random_immigrant_rate * self.population_size))

    def initialize_population(self, genome_factory=None, programme_length: int = 10) -> None:
        """Initialize a random population"""
        self.population.clear()

        for _ in range(self.population_size):
            if genome_factory is not None:
                genome = genome_factory()
            else:
                # Default: create EM43Genome
                rule_array = self.rng.integers(0, 4, 64, dtype=np.uint8)
                rule = EM43Rule(rule_array)
                programme = self.rng.choice([0, 1, 2], size=programme_length, p=[0.7, 0.2, 0.1])
                genome = EM43Genome(rule, programme)

            self.population.append(genome)

    def step(self, fitness_scores: List[float]) -> None:
        """Perform one generation of evolution"""
        if len(fitness_scores) != len(self.population):
            raise ValueError("Number of fitness scores must match population size")

        self.fitness_scores = fitness_scores.copy()

        # Update fitness in genomes
        for genome, fitness in zip(self.population, fitness_scores):
            genome.fitness = fitness

        # Sort population by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        self.population = [self.population[i] for i in sorted_indices]
        self.fitness_scores = [fitness_scores[i] for i in sorted_indices]

        # Create next generation
        next_population = []

        # Elitism: keep best individuals
        for i in range(self.n_elite):
            if isinstance(self.population[i], EM43Genome):
                next_population.append(self.population[i].clone())
            else:
                # For generic genomes, assume they have a clone method
                next_population.append(self.population[i])

        # Generate offspring through tournament selection and crossover
        while len(next_population) < self.population_size:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            child = self._crossover(parent1, parent2)
            child = self._mutate(child)

            next_population.append(child)

        # Random immigrants (replace some non-elite individuals)
        for _ in range(self.n_immigrants):
            if len(next_population) > self.n_elite:
                idx = self.rng.integers(self.n_elite, len(next_population))
                next_population[idx] = self._create_random_genome()

        self.population = next_population[:self.population_size]
        self.generation += 1

    def _tournament_selection(self) -> Union[Genome, EM43Genome]:
        """Tournament selection"""
        tournament_indices = self.rng.choice(
            len(self.population),
            size=min(self.tournament_size, len(self.population)),
            replace=False
        )

        best_idx = tournament_indices[0]
        best_fitness = self.fitness_scores[best_idx]

        for idx in tournament_indices[1:]:
            if self.fitness_scores[idx] > best_fitness:
                best_fitness = self.fitness_scores[idx]
                best_idx = idx

        return self.population[best_idx]

    def _crossover(self, parent1: Union[Genome, EM43Genome], parent2: Union[Genome, EM43Genome]) -> Union[Genome, EM43Genome]:
        """Crossover between two parents"""
        if isinstance(parent1, EM43Genome) and isinstance(parent2, EM43Genome):
            return self._crossover_em43(parent1, parent2)
        else:
            # For generic genomes, return a copy of parent1
            if hasattr(parent1, 'clone'):
                return parent1.clone()
            else:
                return parent1

    def _crossover_em43(self, parent1: EM43Genome, parent2: EM43Genome) -> EM43Genome:
        """Crossover for EM43Genome"""
        # Crossover rules
        child_rule = parent1.rule.crossover(parent2.rule, self.rng)

        # Crossover programmes
        cut_point = self.rng.integers(1, len(parent1.programme))
        child_programme = np.concatenate([
            parent1.programme[:cut_point],
            parent2.programme[cut_point:]
        ])

        return EM43Genome(child_rule, child_programme)

    def _mutate(self, genome: Union[Genome, EM43Genome]) -> Union[Genome, EM43Genome]:
        """Mutate a genome"""
        if isinstance(genome, EM43Genome):
            return self._mutate_em43(genome)
        else:
            return genome

    def _mutate_em43(self, genome: EM43Genome) -> EM43Genome:
        """Mutate an EM43Genome"""
        # Mutate rule
        mutated_rule = genome.rule.mutate(self.mutation_rate, self.rng)

        # Clone and mutate programme
        new_genome = EM43Genome(mutated_rule, genome.programme.copy())
        new_genome.mutate_programme(self.programme_mutation_rate, self.rng)

        return new_genome

    def _create_random_genome(self) -> Union[Genome, EM43Genome]:
        """Create a random genome (for immigrants)"""
        if len(self.population) > 0 and isinstance(self.population[0], EM43Genome):
            # Create random EM43Genome
            rule_array = self.rng.integers(0, 4, 64, dtype=np.uint8)
            rule = EM43Rule(rule_array)
            programme_length = len(self.population[0].programme)
            programme = self.rng.choice([0, 1, 2], size=programme_length, p=[0.7, 0.2, 0.1])
            return EM43Genome(rule, programme)
        else:
            # Fallback - this should be overridden for specific genome types
            raise NotImplementedError("Random genome creation not implemented for this genome type")

    def get_best_genome(self) -> Union[Genome, EM43Genome]:
        """Get the best genome from current population"""
        if not self.population:
            raise ValueError("Population is empty")

        if self.fitness_scores:
            best_idx = np.argmax(self.fitness_scores)
            return self.population[best_idx]
        else:
            return self.population[0]

    def get_population_stats(self) -> dict:
        """Get statistics about the current population"""
        if not self.fitness_scores:
            return {}

        return {
            'generation': self.generation,
            'best_fitness': max(self.fitness_scores),
            'mean_fitness': np.mean(self.fitness_scores),
            'std_fitness': np.std(self.fitness_scores),
            'population_size': len(self.population)
        }
