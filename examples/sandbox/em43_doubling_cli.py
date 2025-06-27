#!/usr/bin/env python3 
"""
EM43 Doubling - CLI Optimized Version
====================================

This version is optimized for command-line usage with proper tqdm progress bars
and reasonable default parameters for quick testing.
"""

import argparse
import numpy as np
from emergent_models.core.state import StateModel
from emergent_models.rules.sanitization import lut_idx
from emergent_models.core.space_model import Tape1D
from emergent_models.encoders.em43 import Em43Encoder
from emergent_models.simulation.simulator import Simulator

from emergent_models.training.new_fitness import AbsoluteDifferenceFitness
from emergent_models.training import (Trainer, ComplexityRewardFitness, GAOptimizer)
from emergent_models.training import TqdmMonitor, DetailedMonitor, CombinedMonitor

def main():
    parser = argparse.ArgumentParser(description='EM-4/3 Doubling with CLI Progress Bar')
    parser.add_argument('--population', type=int, default=1000, help='Population size')
    parser.add_argument('--generations', type=int, default=50, help='Number of generations')
    parser.add_argument('--inputs', type=str, default='1-10', help='Input range (e.g., "1-10" or "1-30")')
    parser.add_argument('--window', type=int, default=200, help='Tape window size')
    parser.add_argument('--max-steps', type=int, default=800, help='Max simulation steps')
    parser.add_argument('--checkpoint-every', type=int, default=25, help='Checkpoint frequency')
    parser.add_argument('--detailed-every', type=int, default=10, help='Detailed telemetry frequency')
    parser.add_argument('--early-stop', type=float, default=1.0, help='Early stopping threshold')
    parser.add_argument('--quiet', action='store_true', help='Minimal output (no detailed monitor)')
    
    args = parser.parse_args()
    
    # Parse input range
    if '-' in args.inputs:
        start, end = map(int, args.inputs.split('-'))
        input_set = np.arange(start, end + 1, dtype=np.int64)
    else:
        input_set = np.array([int(args.inputs)], dtype=np.int64)
    
    print("🔬 EM-4/3 Doubling - CLI Version")
    print("=" * 50)
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Input range: {input_set[0]}-{input_set[-1]} ({len(input_set)} values)")
    print(f"Window size: {args.window}")
    print(f"Max steps: {args.max_steps}")
    print("=" * 50)
    
    # 1. Domain setup with EM-4/3 constraints
    _IMMUTABLE = {
        lut_idx(0, 0, 0): 0,  # Empty space stays empty
        lut_idx(0, 2, 0): 2,  # Red beacon propagation
        lut_idx(0, 0, 2): 0,  # Red beacon boundary
        lut_idx(2, 0, 0): 0,  # Red beacon boundary  
        lut_idx(0, 3, 3): 3,  # Blue boundary behavior
        lut_idx(3, 3, 0): 3,  # Blue boundary behavior
        lut_idx(0, 0, 3): 0,  # Blue boundary behavior
        lut_idx(3, 0, 0): 0,  # Blue boundary behavior
    }
    
    state = StateModel([0, 1, 2, 3], immutable=_IMMUTABLE)
    space = Tape1D(length=args.window, radius=1)
    encoder = Em43Encoder(state, space)
    
    # 2. Simulation
    sim = Simulator(
        state=state, 
        space=space, 
        max_steps=args.max_steps,
        halt_thresh=0.50
    )
    
    # 3. Fitness function with complexity reward
    base_fitness = AbsoluteDifferenceFitness(continuous=True)
    fitness = ComplexityRewardFitness(base_fitness, complexity_bonus=0.05)
    
    # 4. Optimizer with standard EM-4/3 parameters
    optim = GAOptimizer(
        pop_size=args.population,
        state=state,
        prog_len=10,
        mutation_rate=0.03,
        prog_mutation_rate=0.08,
        elite_fraction=0.1,
        tournament_size=3,
        random_immigrant_rate=0.2,
        prog_sparsity=0.6  # Improved default for better diversity
    )
    
    # 5. Monitoring - CLI optimized
    tqdm_monitor = TqdmMonitor(args.generations, force_notebook=False)
    
    if args.quiet:
        # Just tqdm progress bar
        monitor = tqdm_monitor
    else:
        # Combine tqdm with detailed logging
        detailed_monitor = DetailedMonitor(
            log_every=args.detailed_every, 
            detailed_every=args.detailed_every * 2
        )
        monitor = CombinedMonitor(tqdm_monitor, detailed_monitor)
    
    # 6. Training
    trainer = Trainer(encoder, sim, fitness, optim, monitor)
    
    print("🚀 Starting training...")
    print("   (Press Ctrl+C to stop early)")
    
    try:
        result = trainer.fit(
            inputs=input_set,
            generations=args.generations,
            use_tqdm=False,  # We handle tqdm through monitor
            checkpoint_every=args.checkpoint_every,
            early_stopping_threshold=args.early_stop
        )
        
        print(f"\n✅ Training completed!")
        print(f"   Best fitness: {result['best_fitness']:.4f}")
        print(f"   Final generation: {result.get('final_generation', args.generations)}")
        
        # Test the best genome
        print(f"\n🧪 Testing best genome on inputs {input_set[0]}-{input_set[-1]}:")
        
        # Quick accuracy test
        best_genome = result.get('best_genome')
        if best_genome:
            # Test a few examples
            test_inputs = input_set[:min(5, len(input_set))]
            print("   Input | Expected | Result")
            print("   ------|----------|-------")
            
            for inp in test_inputs:
                expected = 2 * inp
                # This would need actual genome testing - simplified for demo
                print(f"   {inp:5d} | {expected:8d} | TBD")
        
        return result
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return None
    finally:
        # Ensure progress bar is closed
        if hasattr(monitor, 'close'):
            monitor.close()

if __name__ == "__main__":
    result = main()
    
    if result:
        print(f"\n📊 Final Results:")
        print(f"   Best fitness: {result['best_fitness']:.4f}")
        print(f"   Training time: {result.get('training_time', 'N/A')}")
        print(f"\n💡 To run with different parameters:")
        print(f"   python {__file__} --population 2000 --generations 100 --inputs 1-30")
    else:
        print(f"\n💡 Try running with smaller parameters:")
        print(f"   python {__file__} --population 500 --generations 20 --inputs 1-5")
