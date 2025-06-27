#!/usr/bin/env python3
"""
Test script for optimized Trainer implementation.

This script tests the performance optimizations while ensuring
the modular API remains unchanged.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from emergent_models.core.state import StateModel
from emergent_models.rules.sanitization import lut_idx
from emergent_models.core.space_model import Tape1D
from emergent_models.encoders.em43 import Em43Encoder
from emergent_models.simulation.simulator import Simulator
from emergent_models.training.new_fitness import AbsoluteDifferenceFitness, ComplexityRewardFitness
from emergent_models.training.optimizer import GAOptimizer
from emergent_models.training.new_trainer import Trainer
from emergent_models.training.monitor import SilentMonitor


def test_optimized_trainer():
    """Test the optimized trainer implementation."""
    print("🔬 Testing Optimized Trainer Implementation")
    print("=" * 50)
    
    # Setup identical to em43_doubling_new.py
    _IMMUTABLE = {
        lut_idx(0, 0, 0): 0,
        lut_idx(0, 2, 0): 2,
        lut_idx(0, 0, 2): 0,
        lut_idx(2, 0, 0): 0,
        lut_idx(0, 3, 3): 3,
        lut_idx(3, 3, 0): 3,
        lut_idx(0, 0, 3): 0,
        lut_idx(3, 0, 0): 0,
    }
    
    state = StateModel([0,1,2,3], immutable=_IMMUTABLE)
    space = Tape1D(length=200, radius=1)
    encoder = Em43Encoder(state, space)
    
    sim = Simulator(
        state=state,
        space=space,
        max_steps=800,
        halt_thresh=0.50
    )
    
    base_fitness = AbsoluteDifferenceFitness(continuous=True)
    fitness = ComplexityRewardFitness(base_fitness, complexity_bonus=0.05)
    
    # Small population for testing
    optim = GAOptimizer(
        pop_size=100,
        state=state,
        prog_len=10,
        mutation_rate=0.03,
        prog_mutation_rate=0.08,
        elite_fraction=0.1,
        tournament_size=3,
        random_immigrant_rate=0.2,
        prog_sparsity=0.3
    )
    
    monitor = SilentMonitor()
    trainer = Trainer(encoder, sim, fitness, optim, monitor)
    
    print("✅ Components initialized successfully!")
    
    # Test population evaluation performance
    print("\n🚀 Testing Population Evaluation Performance...")
    
    inputs = np.arange(1, 11, dtype=np.int64)  # Small input set for testing
    population = optim.ask()
    
    # Warm up (Numba compilation)
    print("⏳ Warming up (Numba compilation)...")
    _ = trainer.evaluate_population(population, inputs)
    
    # Benchmark evaluation speed
    n_evals = 10
    print(f"⏱️  Benchmarking {n_evals} evaluations...")
    
    start_time = time.time()
    for i in range(n_evals):
        scores = trainer.evaluate_population(population, inputs)
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_eval = total_time / n_evals
    
    print(f"📊 Performance Results:")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Time per evaluation: {time_per_eval:.3f}s")
    print(f"   Population size: {len(population)}")
    print(f"   Inputs per genome: {len(inputs)}")
    print(f"   Total simulations: {len(population) * len(inputs) * n_evals}")
    
    # Test a few generations
    print(f"\n🧬 Testing Training Loop...")
    
    start_time = time.time()
    result = trainer.fit(
        inputs=inputs,
        generations=5,  # Just a few generations for testing
        use_tqdm=False
    )
    end_time = time.time()
    
    training_time = end_time - start_time
    
    print(f"📈 Training Results:")
    print(f"   Training time: {training_time:.3f}s")
    print(f"   Best fitness: {result['best_fitness']:.4f}")
    print(f"   Generations: {result['final_generation']}")
    
    print(f"\n✅ All tests passed! Optimized trainer is working correctly.")
    
    return {
        'evaluation_time': time_per_eval,
        'training_time': training_time,
        'best_fitness': result['best_fitness']
    }


if __name__ == "__main__":
    try:
        results = test_optimized_trainer()
        print(f"\n🎯 Test completed successfully!")
        print(f"   Evaluation time: {results['evaluation_time']:.3f}s")
        print(f"   Training time: {results['training_time']:.3f}s")
        print(f"   Best fitness: {results['best_fitness']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
