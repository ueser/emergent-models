from __future__ import annotations

from typing import List, Union, Optional, Callable
import numpy as np
from tqdm import tqdm
import time

from ..core.genome import Genome
from ..rules.em43 import EM43Genome
from ..losses.base import CALoss
from ..optimizers.genetic import GAOptimizer
from ..simulators.base import Simulator
from ..simulators.numba_simulator import NumbaSimulator
from ..data.dataloader import CADataLoader


def create_accuracy_validator(test_inputs: List[int], expected_outputs: List[int]):
    """
    Create a validation function that computes accuracy on a test set.

    Parameters
    ----------
    test_inputs : List[int]
        Input values to test
    expected_outputs : List[int]
        Expected output values

    Returns
    -------
    Callable
        Validation function that takes (genome, simulator) and returns accuracy
    """
    def validate(genome, simulator):
        try:
            if hasattr(simulator, 'simulate_batch'):
                outputs = simulator.simulate_batch(genome, test_inputs)
            else:
                # Fallback for non-batch simulators
                outputs = []
                for inp in test_inputs:
                    from ..core.space import Space1D
                    input_space = Space1D(1, n_states=64)
                    input_space.data[0] = inp
                    result = simulator(genome, input_space)
                    outputs.append(result.data[0] if len(result.data) > 0 else -1)

            # Calculate accuracy
            correct = sum(1 for out, exp in zip(outputs, expected_outputs)
                         if out >= 0 and out == exp)
            accuracy = correct / len(test_inputs)
            return accuracy

        except Exception as e:
            print(f"Validation error: {e}")
            return 0.0

    return validate


class CATrainer:
    """High-level training interface for cellular automata evolution."""

    def __init__(
        self,
        simulator: Simulator,
        optimizer: GAOptimizer,
        loss_fn: CALoss,
        device: str = "cpu",
        verbose: bool = True
    ) -> None:
        self.simulator = simulator
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.verbose = verbose

        # Training history
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'generation_times': []
        }

        self.best_genome: Optional[Union[Genome, EM43Genome]] = None
        self.best_fitness = float('-inf')

    def fit(
        self,
        population: Optional[List[Union[Genome, EM43Genome]]] = None,
        dataloader: Optional[CADataLoader] = None,
        epochs: int = 100,
        fitness_fn: Optional[Callable] = None,
        checkpoint_every: int = 50,
        checkpoint_dir: str = "checkpoints",
        early_stopping_threshold: Optional[float] = None,
        early_stopping_metric: str = "fitness",
        validation_fn: Optional[Callable] = None
    ) -> None:
        """
        Train the population using genetic algorithm.

        Parameters
        ----------
        population : List of genomes (optional if optimizer has population)
        dataloader : DataLoader for training data
        epochs : Number of generations to train
        fitness_fn : Custom fitness function (overrides dataloader evaluation)
        checkpoint_every : Save checkpoint every N generations
        checkpoint_dir : Directory to save checkpoints
        early_stopping_threshold : Stop training if metric reaches this value (e.g., 1.0 for 100% accuracy)
        early_stopping_metric : Metric to monitor ("fitness", "accuracy", "validation")
        validation_fn : Function to compute validation accuracy for early stopping
        """

        # Initialize population if needed
        if population is not None:
            self.optimizer.population = population
        elif not self.optimizer.population:
            raise ValueError("No population provided and optimizer has no population")

        # Setup progress bar
        pbar = tqdm(range(epochs), desc="Training", disable=not self.verbose)

        for epoch in pbar:
            start_time = time.time()

            # Evaluate fitness
            if fitness_fn is not None:
                fitness_scores = self._evaluate_with_function(fitness_fn)
            elif dataloader is not None:
                fitness_scores = self._evaluate_with_dataloader(dataloader)
            else:
                raise ValueError("Either fitness_fn or dataloader must be provided")

            # Update best genome
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[max_fitness_idx]
                self.best_genome = self.optimizer.population[max_fitness_idx]
                if hasattr(self.best_genome, 'clone'):
                    self.best_genome = self.best_genome.clone()

            # Record history
            self.history['best_fitness'].append(np.max(fitness_scores))
            self.history['mean_fitness'].append(np.mean(fitness_scores))
            self.history['std_fitness'].append(np.std(fitness_scores))

            # Evolution step
            self.optimizer.step(fitness_scores)

            generation_time = time.time() - start_time
            self.history['generation_times'].append(generation_time)

            # Early stopping check
            should_stop = False
            stopping_metric_value = None

            if early_stopping_threshold is not None:
                if early_stopping_metric == "fitness":
                    stopping_metric_value = np.max(fitness_scores)
                    should_stop = stopping_metric_value >= early_stopping_threshold

                elif early_stopping_metric == "accuracy" and validation_fn is not None:
                    stopping_metric_value = validation_fn(self.best_genome, self.simulator)
                    should_stop = stopping_metric_value >= early_stopping_threshold

                elif early_stopping_metric == "validation" and validation_fn is not None:
                    stopping_metric_value = validation_fn(self.best_genome, self.simulator)
                    should_stop = stopping_metric_value >= early_stopping_threshold

            # Update progress bar
            if self.verbose:
                postfix = {
                    'best': f"{np.max(fitness_scores):.3f}",
                    'mean': f"{np.mean(fitness_scores):.3f}",
                    'time': f"{generation_time:.2f}s"
                }
                if stopping_metric_value is not None:
                    postfix[early_stopping_metric] = f"{stopping_metric_value:.3f}"
                pbar.set_postfix(postfix)

            # Early stopping
            if should_stop:
                if self.verbose:
                    print(f"\nEarly stopping at generation {epoch + 1}")
                    print(f"{early_stopping_metric.capitalize()} reached threshold: {stopping_metric_value:.4f} >= {early_stopping_threshold}")

                # Save final checkpoint
                self._save_checkpoint(epoch + 1, checkpoint_dir, final=True)
                break

            # Regular checkpointing
            if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
                self._save_checkpoint(epoch + 1, checkpoint_dir)

    def _evaluate_with_dataloader(self, dataloader: CADataLoader) -> List[float]:
        """Evaluate population fitness using a dataloader"""
        fitness_scores = []

        for genome in self.optimizer.population:
            total_loss = 0.0
            total_samples = 0

            for batch in dataloader:
                for input_state, target_state in batch:
                    output = self.simulator(genome, input_state)
                    loss = self.loss_fn(output, target_state)
                    total_loss += loss
                    total_samples += 1

            # Convert loss to fitness (negative loss)
            avg_loss = total_loss / max(total_samples, 1)
            fitness = -avg_loss  # Higher fitness is better
            fitness_scores.append(fitness)

        return fitness_scores

    def _evaluate_with_function(self, fitness_fn: Callable) -> List[float]:
        """Evaluate population fitness using a custom function"""
        fitness_scores = []

        for genome in self.optimizer.population:
            fitness = fitness_fn(genome, self.simulator)
            fitness_scores.append(fitness)

        return fitness_scores

    def _save_checkpoint(self, generation: int, checkpoint_dir: str, final: bool = False) -> None:
        """Save training checkpoint"""
        import os
        import pickle

        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'generation': generation,
            'best_genome': self.best_genome,
            'best_fitness': self.best_fitness,
            'history': self.history,
            'optimizer_state': {
                'generation': self.optimizer.generation,
                'population_size': self.optimizer.population_size
            },
            'early_stopped': final
        }

        if final:
            checkpoint_path = os.path.join(checkpoint_dir, f"final_checkpoint_gen_{generation:04d}.pkl")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_gen_{generation:04d}.pkl")

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        if self.verbose:
            status = "final " if final else ""
            print(f"Saved {status}checkpoint: {checkpoint_path}")

    def evaluate_genome(
        self,
        genome: Union[Genome, EM43Genome],
        dataloader: CADataLoader
    ) -> float:
        """Evaluate a single genome"""
        total_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            for input_state, target_state in batch:
                output = self.simulator(genome, input_state)
                loss = self.loss_fn(output, target_state)
                total_loss += loss
                total_samples += 1

        return total_loss / max(total_samples, 1)

    def get_training_stats(self) -> dict:
        """Get training statistics"""
        if not self.history['best_fitness']:
            return {}

        return {
            'generations': len(self.history['best_fitness']),
            'best_fitness': max(self.history['best_fitness']),
            'final_mean_fitness': self.history['mean_fitness'][-1],
            'total_time': sum(self.history['generation_times']),
            'avg_generation_time': np.mean(self.history['generation_times'])
        }
