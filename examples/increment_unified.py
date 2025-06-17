#!/usr/bin/env python3
"""
EM-4/3 Unified Increment Example
================================
Train a cellular automaton to perform the increment operation (x -> x+1)
with configurable encoding (binary or positional).

This demonstrates the proper modular architecture:
1. Encoder encodes inputs + programme → initial space state
2. Simulator simulates initial state + rules → final space state  
3. Encoder decodes final state → output
4. Fitness calculated based on decoded output

Usage:
    python examples/increment_unified.py --encoding binary
    python examples/increment_unified.py --encoding positional
"""

import argparse
import numpy as np
import time
from pathlib import Path

# Import emergent_models components
from emergent_models.rules.em43 import EM43Rule, EM43Genome
from emergent_models.simulators.numba_simulator import NumbaSimulator
from emergent_models.optimizers.genetic import GAOptimizer
from emergent_models.losses.distance import HammingLoss
from emergent_models.training.trainer import CATrainer
from emergent_models.utils.visualization import plot_fitness_curve
from emergent_models.training.checkpointing import save_genome, load_genome
from emergent_models.encoders.binary import EM43BinaryEncoder
from emergent_models.encoders.position import EM43PositionalEncoder


def create_unified_fitness_function(simulator: NumbaSimulator, encoder, input_range=(1, 15)):
    """Create a fitness function that works with any encoder"""
    inputs = list(range(input_range[0], input_range[1] + 1))
    targets = [x + 1 for x in inputs]
    
    def fitness_fn(genome: EM43Genome, sim: NumbaSimulator) -> float:
        """Evaluate genome fitness using the modular architecture"""
        try:
            correct = 0
            total = len(inputs)
            
            for input_val, target_val in zip(inputs, targets):
                # 1. Encoder creates initial space
                initial_space = encoder.encode_input(
                    genome.programme, input_val, sim.window
                )
                
                # 2. Simulator simulates to final space
                final_spaces = sim.simulate_spaces(genome, [initial_space])
                final_space = final_spaces[0]
                
                # 3. Encoder decodes output from final space
                if isinstance(encoder, EM43BinaryEncoder):
                    output_val = encoder.decode_output(final_space, len(genome.programme))
                else:  # EM43PositionalEncoder
                    output_val = encoder.decode_output(final_space, len(genome.programme), input_val)
                
                # 4. Check if output matches target
                if output_val >= 0 and output_val == target_val:
                    correct += 1
            
            accuracy = correct / total
            
            # Add sparsity penalty
            sparsity_penalty = 0.01 * np.count_nonzero(genome.programme) / len(genome.programme)
            
            return accuracy - sparsity_penalty
            
        except Exception as e:
            print(f"Error evaluating genome: {e}")
            return -1.0
    
    return fitness_fn


def create_unified_validation_function(encoder, test_inputs, test_targets, simulator):
    """Create validation function that works with any encoder"""
    def validate(genome, sim):
        try:
            correct = 0
            
            for input_val, target_val in zip(test_inputs, test_targets):
                # Use the same modular workflow
                initial_space = encoder.encode_input(
                    genome.programme, input_val, sim.window
                )
                final_spaces = sim.simulate_spaces(genome, [initial_space])
                final_space = final_spaces[0]
                
                if isinstance(encoder, EM43BinaryEncoder):
                    output_val = encoder.decode_output(final_space, len(genome.programme))
                else:  # EM43PositionalEncoder
                    output_val = encoder.decode_output(final_space, len(genome.programme), input_val)
                
                if output_val >= 0 and output_val == target_val:
                    correct += 1
            
            return correct / len(test_inputs)
            
        except Exception as e:
            print(f"Validation error: {e}")
            return 0.0
    
    return validate


def main():
    """Main training loop"""
    parser = argparse.ArgumentParser(description='Train EM-4/3 increment with configurable encoding')
    parser.add_argument('--encoding', choices=['binary', 'positional'], default='binary',
                       help='Encoding method to use')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations')
    parser.add_argument('--population', type=int, default=300, help='Population size')
    
    args = parser.parse_args()
    
    print(f"EM-4/3 Increment Task Training ({args.encoding.upper()} encoding)")
    print("=" * 60)
    
    # Create encoder based on choice
    if args.encoding == 'binary':
        encoder = EM43BinaryEncoder(bit_width=8, input_state=2)
        print("Using binary encoding: numbers → binary → state mapping")
        print("Example: 5 → 00000101 → [0,0,0,0,0,2,0,2]")
    else:
        encoder = EM43PositionalEncoder(beacon_state=2, separator_state=3)
        print("Using positional encoding: numbers → positional → beacon placement")
        print("Example: 5 → [0,0,0,0,0,2] (5 zeros + red beacon)")
    
    # Configuration
    config = {
        'population_size': args.population,
        'generations': args.generations,
        'programme_length': 8 if args.encoding == 'binary' else 6,
        'window_size': 120 if args.encoding == 'binary' else 200,
        'max_steps': 150,
        'halt_thresh': 0.50,
        'mutation_rate': 0.05,
        'programme_mutation_rate': 0.10,
        'elite_fraction': 0.15,
        'tournament_size': 3,
        'random_immigrant_rate': 0.25,
        'input_range': (1, 15) if args.encoding == 'binary' else (1, 20)
    }
    
    print(f"\nConfiguration:")
    print(f"- Population size: {config['population_size']}")
    print(f"- Generations: {config['generations']}")
    print(f"- Programme length: {config['programme_length']}")
    print(f"- Window size: {config['window_size']}")
    print(f"- Input range: {config['input_range']}")
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
    fitness_fn = create_unified_fitness_function(
        simulator, encoder, config['input_range']
    )
    
    # Create validation function for early stopping
    test_inputs = list(range(1, 8))
    test_targets = [x + 1 for x in test_inputs]
    validation_fn = create_unified_validation_function(
        encoder, test_inputs, test_targets, simulator
    )
    
    # Create trainer
    trainer = CATrainer(simulator, optimizer, HammingLoss(), verbose=True)
    
    # Training with early stopping
    print("Starting training...")
    print("Early stopping enabled: will stop if 100% accuracy is reached")
    start_time = time.time()
    
    trainer.fit(
        fitness_fn=fitness_fn,
        epochs=config['generations'],
        checkpoint_every=25,
        early_stopping_threshold=1.0,
        early_stopping_metric="accuracy",
        validation_fn=validation_fn
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Test the best genome
    best_genome = trainer.best_genome
    print(f"\nBest fitness: {trainer.best_fitness:.4f}")
    print(f"Best genome programme: {best_genome.programme}")
    
    # Test on full range
    print(f"\nTesting best genome with {args.encoding} encoding:")
    test_inputs_full = list(range(1, 16))
    
    print("Input -> Expected -> Actual -> Status")
    print("-" * 35)
    
    correct = 0
    for inp in test_inputs_full:
        expected = inp + 1
        
        try:
            # Use modular architecture for testing
            initial_space = encoder.encode_input(best_genome.programme, inp, simulator.window)
            final_spaces = simulator.simulate_spaces(best_genome, [initial_space])
            final_space = final_spaces[0]
            
            if isinstance(encoder, EM43BinaryEncoder):
                out = encoder.decode_output(final_space, len(best_genome.programme))
            else:
                out = encoder.decode_output(final_space, len(best_genome.programme), inp)
            
            status = "✓" if out == expected else "✗"
            if out == expected:
                correct += 1
            print(f"{inp:2d} -> {expected:2d} -> {out:2d} {status}")
            
        except Exception as e:
            print(f"{inp:2d} -> {expected:2d} -> ERR {e}")
    
    accuracy = correct / len(test_inputs_full)
    print(f"\nAccuracy: {accuracy:.2%}")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    filename = f"best_{args.encoding}_increment_genome.json"
    save_genome(best_genome, results_dir / filename)
    print(f"Saved best genome to {results_dir / filename}")
    
    # Plot fitness curve
    plot_filename = f"{args.encoding}_increment_fitness_curve.png"
    plot_fitness_curve(trainer.history, save_path=results_dir / plot_filename)
    print(f"Saved fitness curve to {results_dir / plot_filename}")
    
    return best_genome, trainer.history


if __name__ == "__main__":
    main()
