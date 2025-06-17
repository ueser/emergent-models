#!/usr/bin/env python3
"""
EM-4/3 Increment Example
========================
Train a cellular automaton to perform the increment operation (x -> x+1)
using the EM-4/3 system with genetic algorithm optimization.

This example demonstrates:
- Creating EM-4/3 genomes for increment task
- Setting up batch simulation with Numba acceleration
- Training with genetic algorithm and early stopping
- Evaluating and visualizing results

Usage:
    python examples/increment.py
"""

import numpy as np
import time
from pathlib import Path

# Import emergent_models components
from emergent_models.rules.em43 import EM43Rule, EM43Genome
from emergent_models.simulators.numba_simulator import NumbaSimulator
from emergent_models.optimizers.genetic import GAOptimizer
from emergent_models.data.mathematical import IncrementDataset
from emergent_models.data.dataloader import CADataLoader
from emergent_models.losses.distance import HammingLoss
from emergent_models.training.trainer import CATrainer, create_accuracy_validator
from emergent_models.utils.visualization import plot_fitness_curve
from emergent_models.training.checkpointing import save_genome, load_genome


def create_fitness_function(simulator: NumbaSimulator, input_range=(1, 32)):
    """Create a fitness function for the increment task"""
    inputs = list(range(input_range[0], input_range[1] + 1))
    targets = [x + 1 for x in inputs]

    def fitness_fn(genome: EM43Genome, sim: NumbaSimulator) -> float:
        """Evaluate genome fitness on increment task"""
        try:
            outputs = sim.simulate_batch(genome, inputs)

            # Calculate accuracy
            correct = 0
            total = len(inputs)

            for i, (target, output) in enumerate(zip(targets, outputs)):
                if output >= 0 and output == target:
                    correct += 1

            accuracy = correct / total

            # Add sparsity penalty (encourage simpler programmes)
            sparsity_penalty = 0.01 * np.count_nonzero(genome.programme) / len(genome.programme)

            # Fitness is accuracy minus sparsity penalty
            fitness = accuracy - sparsity_penalty

            return fitness

        except Exception as e:
            print(f"Error evaluating genome: {e}")
            return -1.0

    return fitness_fn


def main():
    """Main training loop"""
    print("EM-4/3 Increment Task Training")
    print("=" * 40)

    # Configuration
    config = {
        'population_size': 500,
        'generations': 150,
        'programme_length': 8,
        'window_size': 150,
        'max_steps': 200,
        'halt_thresh': 0.50,
        'mutation_rate': 0.03,
        'programme_mutation_rate': 0.08,
        'elite_fraction': 0.1,
        'tournament_size': 3,
        'random_immigrant_rate': 0.2
    }

    print(f"Population size: {config['population_size']}")
    print(f"Generations: {config['generations']}")
    print(f"Programme length: {config['programme_length']}")
    print()

    # Create simulator
    simulator = NumbaSimulator(
        window=config['window_size'],
        max_steps=config['max_steps'],
        halt_thresh=config['halt_thresh']
    )

    # Create optimizer
    optimizer = GAOptimizer(
        population_size=config['population_size'],
        mutation_rate=config['mutation_rate'],
        programme_mutation_rate=config['programme_mutation_rate'],
        elite_fraction=config['elite_fraction'],
        tournament_size=config['tournament_size'],
        random_immigrant_rate=config['random_immigrant_rate']
    )

    # Initialize population
    print("Initializing population...")
    optimizer.initialize_population(programme_length=config['programme_length'])

    # Create fitness function
    fitness_fn = create_fitness_function(simulator)

    # Create validation function for early stopping
    test_inputs = list(range(1, 11))  # Test on inputs 1-10
    test_targets = [x + 1 for x in test_inputs]  # Expected outputs
    validation_fn = create_accuracy_validator(test_inputs, test_targets)

    # Create trainer
    trainer = CATrainer(simulator, optimizer, HammingLoss(), verbose=True)

    # Training with early stopping
    print("Starting training...")
    print("Early stopping enabled: will stop if 100% accuracy is reached on test set")
    start_time = time.time()

    trainer.fit(
        fitness_fn=fitness_fn,
        epochs=config['generations'],
        checkpoint_every=50,
        early_stopping_threshold=1.0,  # 100% accuracy
        early_stopping_metric="accuracy",
        validation_fn=validation_fn
    )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Results
    best_genome = trainer.best_genome
    best_fitness = trainer.best_fitness

    print(f"\nBest fitness: {best_fitness:.4f}")
    print(f"Best genome programme: {best_genome.programme}")

    # Test the best genome
    print("\nTesting best genome:")
    test_inputs = list(range(1, 21))
    test_outputs = simulator.simulate_batch(best_genome, test_inputs)

    print("Input -> Expected -> Actual")
    print("-" * 30)
    for inp, out in zip(test_inputs, test_outputs):
        expected = inp + 1
        status = "✓" if out == expected else "✗"
        print(f"{inp:2d} -> {expected:2d} -> {out:2d} {status}")

    # Calculate final accuracy
    correct = sum(1 for inp, out in zip(test_inputs, test_outputs) if out == inp + 1)
    accuracy = correct / len(test_inputs)
    print(f"\nAccuracy on test set: {accuracy:.2%}")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save best genome
    save_genome(best_genome, results_dir / "best_increment_genome.json")
    print(f"Saved best genome to {results_dir / 'best_increment_genome.json'}")

    # Plot fitness curve
    plot_fitness_curve(trainer.history, save_path=results_dir / "increment_fitness_curve.png")
    print(f"Saved fitness curve to {results_dir / 'increment_fitness_curve.png'}")

    # Print training statistics
    stats = trainer.get_training_stats()
    print(f"\nTraining Statistics:")
    print(f"- Total generations: {stats['generations']}")
    print(f"- Best fitness achieved: {stats['best_fitness']:.4f}")
    print(f"- Final mean fitness: {stats['final_mean_fitness']:.4f}")
    print(f"- Average time per generation: {stats['avg_generation_time']:.2f}s")

    return best_genome, trainer.history


def test_saved_genome():
    """Test a previously saved genome"""
    try:
        genome = load_genome("results/best_increment_genome.json")
        print(f"Loaded genome with fitness: {genome.fitness:.4f}")

        simulator = NumbaSimulator(window=150, max_steps=200, halt_thresh=0.50)

        # Test on increment task
        test_inputs = list(range(1, 21))
        outputs = simulator.simulate_batch(genome, test_inputs)

        print("\nTest Results:")
        print("Input -> Expected -> Actual")
        print("-" * 30)

        correct = 0
        for inp, out in zip(test_inputs, outputs):
            expected = inp + 1
            status = "✓" if out == expected else "✗"
            if out == expected:
                correct += 1
            print(f"{inp:2d} -> {expected:2d} -> {out:2d} {status}")

        accuracy = correct / len(test_inputs)
        print(f"\nAccuracy: {accuracy:.2%}")

    except FileNotFoundError:
        print("No saved genome found. Run training first.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_saved_genome()
    else:
        main()
