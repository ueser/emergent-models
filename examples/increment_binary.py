#!/usr/bin/env python3
"""
EM-4/3 Binary Increment Example
===============================
Train a cellular automaton to perform the increment operation (x -> x+1)
using binary encoding instead of positional encoding.

Binary encoding: 9 -> 01001 -> 02002 (where 1's become state 2)

This example demonstrates:
- Binary encoding of inputs for EM-4/3 CA
- Training with genetic algorithm and early stopping
- Comparison with positional encoding approach

Usage:
    python examples/increment_binary.py
"""

import numpy as np
import time
from pathlib import Path

# Import emergent_models components
from emergent_models.rules.em43 import EM43Rule, EM43Genome
from emergent_models.simulators.numba_simulator import NumbaSimulator
from emergent_models.optimizers.genetic import GAOptimizer
from emergent_models.losses.distance import HammingLoss
from emergent_models.training.trainer import CATrainer, create_accuracy_validator
from emergent_models.utils.visualization import plot_fitness_curve
from emergent_models.training.checkpointing import save_genome, load_genome
from emergent_models.encoders.binary import int_to_binary_states, binary_states_to_int


def create_binary_fitness_function(simulator: NumbaSimulator, input_range=(1, 15), bit_width=8):
    """Create a fitness function for the binary increment task"""
    inputs = list(range(input_range[0], input_range[1] + 1))
    targets = [x + 1 for x in inputs]
    
    def fitness_fn(genome: EM43Genome, sim: NumbaSimulator) -> float:
        """Evaluate genome fitness on binary increment task"""
        try:
            outputs = sim.simulate_batch(
                genome, inputs, 
                use_binary_encoding=True, 
                bit_width=bit_width, 
                input_state=2
            )
            
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


def create_binary_validation_function(test_inputs, test_targets, bit_width=8):
    """Create validation function for binary encoding"""
    def validate(genome, simulator):
        try:
            outputs = simulator.simulate_batch(
                genome, test_inputs,
                use_binary_encoding=True,
                bit_width=bit_width,
                input_state=2
            )
            
            # Calculate accuracy
            correct = sum(1 for out, exp in zip(outputs, test_targets) 
                         if out >= 0 and out == exp)
            accuracy = correct / len(test_inputs)
            return accuracy
            
        except Exception as e:
            print(f"Validation error: {e}")
            return 0.0
    
    return validate


def test_binary_encoding():
    """Test the binary encoding functionality"""
    print("Testing binary encoding...")
    
    # Test encoding
    for num in [1, 5, 9, 15]:
        binary_states = int_to_binary_states(num, bit_width=8, input_state=2)
        decoded = binary_states_to_int(binary_states, input_state=2)
        binary_str = ''.join('1' if s == 2 else '0' for s in binary_states)
        print(f"{num:2d} -> {binary_str} -> {binary_states} -> {decoded}")
    
    print()


def main():
    """Main training loop"""
    print("EM-4/3 Binary Increment Task Training")
    print("=" * 45)
    
    # Test binary encoding first
    test_binary_encoding()
    
    # Configuration
    config = {
        'population_size': 300,
        'generations': 100,
        'programme_length': 6,
        'window_size': 120,
        'max_steps': 150,
        'halt_thresh': 0.50,
        'mutation_rate': 0.05,
        'programme_mutation_rate': 0.10,
        'elite_fraction': 0.15,
        'tournament_size': 3,
        'random_immigrant_rate': 0.25,
        'bit_width': 8,
        'input_range': (1, 15)  # Smaller range for binary
    }
    
    print(f"Population size: {config['population_size']}")
    print(f"Generations: {config['generations']}")
    print(f"Programme length: {config['programme_length']}")
    print(f"Binary bit width: {config['bit_width']}")
    print(f"Input range: {config['input_range']}")
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
    fitness_fn = create_binary_fitness_function(
        simulator, 
        config['input_range'], 
        config['bit_width']
    )
    
    # Create validation function for early stopping
    test_inputs = list(range(1, 8))  # Test on inputs 1-7
    test_targets = [x + 1 for x in test_inputs]  # Expected outputs
    validation_fn = create_binary_validation_function(
        test_inputs, test_targets, config['bit_width']
    )
    
    # Create trainer
    trainer = CATrainer(simulator, optimizer, HammingLoss(), verbose=True)
    
    # Training with early stopping
    print("Starting training with binary encoding...")
    print("Early stopping enabled: will stop if 100% accuracy is reached on test set")
    start_time = time.time()
    
    trainer.fit(
        fitness_fn=fitness_fn,
        epochs=config['generations'],
        checkpoint_every=25,
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
    print("\nTesting best genome with binary encoding:")
    test_inputs_full = list(range(1, 16))
    test_outputs = simulator.simulate_batch(
        best_genome, test_inputs_full,
        use_binary_encoding=True,
        bit_width=config['bit_width'],
        input_state=2
    )
    
    print("Input -> Expected -> Actual -> Binary Input -> Binary Output")
    print("-" * 65)
    for inp, out in zip(test_inputs_full, test_outputs):
        expected = inp + 1
        status = "✓" if out == expected else "✗"
        
        # Show binary representations
        inp_binary = int_to_binary_states(inp, config['bit_width'], 2)
        inp_bin_str = ''.join('1' if s == 2 else '0' for s in inp_binary)
        
        if out >= 0:
            out_binary = int_to_binary_states(out, config['bit_width'], 2)
            out_bin_str = ''.join('1' if s == 2 else '0' for s in out_binary)
        else:
            out_bin_str = "FAIL"
        
        print(f"{inp:2d} -> {expected:2d} -> {out:2d} -> {inp_bin_str} -> {out_bin_str} {status}")
    
    # Calculate final accuracy
    correct = sum(1 for inp, out in zip(test_inputs_full, test_outputs) if out == inp + 1)
    accuracy = correct / len(test_inputs_full)
    print(f"\nAccuracy on test set: {accuracy:.2%}")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save best genome
    save_genome(best_genome, results_dir / "best_binary_increment_genome.json")
    print(f"Saved best genome to {results_dir / 'best_binary_increment_genome.json'}")
    
    # Plot fitness curve
    plot_fitness_curve(trainer.history, save_path=results_dir / "binary_increment_fitness_curve.png")
    print(f"Saved fitness curve to {results_dir / 'binary_increment_fitness_curve.png'}")
    
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
        genome = load_genome("results/best_binary_increment_genome.json")
        print(f"Loaded genome with fitness: {genome.fitness:.4f}")
        
        simulator = NumbaSimulator(window=120, max_steps=150, halt_thresh=0.50)
        
        # Test on increment task with binary encoding
        test_inputs = list(range(1, 16))
        outputs = simulator.simulate_batch(
            genome, test_inputs,
            use_binary_encoding=True,
            bit_width=8,
            input_state=2
        )
        
        print("\nTest Results with Binary Encoding:")
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
