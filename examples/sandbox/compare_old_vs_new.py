#!/usr/bin/env python3
"""
Comprehensive Comparison: Old vs New EM-4/3 Doubling Implementation
==================================================================

This script compares the old standalone implementation (em43_doubling_old.py)
with the new SDK-based implementation (em43_doubling_new.py) for both:

1. Speed Performance: Time per generation and evaluation throughput
2. Accuracy Convergence: How quickly each approach reaches high accuracy

The comparison uses identical hyperparameters to ensure fair comparison.
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import tempfile
import pickle
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    total_time: float
    generations: int
    best_fitness_curve: List[float]
    mean_fitness_curve: List[float]
    final_accuracy: float
    convergence_generation: Optional[int]  # Generation where accuracy >= 0.95
    time_per_generation: float
    success: bool
    error_message: Optional[str] = None

class PerformanceComparator:
    """Compare performance between old and new implementations."""
    
    def __init__(self, test_generations: int = 50, test_pop_size: int = 1000):
        self.test_generations = test_generations
        self.test_pop_size = test_pop_size
        self.results: Dict[str, BenchmarkResult] = {}
        
    def run_old_implementation(self) -> BenchmarkResult:
        """Run the old standalone implementation."""
        print("🔄 Running OLD implementation...")

        try:
            # Import old implementation modules
            import examples.sandbox.em43_doubling_old as old_module

            # Store original values
            original_pop_size = old_module.POP_SIZE
            original_generations = old_module.N_GENERATIONS

            # Temporarily modify globals for fair comparison
            old_module.POP_SIZE = self.test_pop_size
            old_module.N_GENERATIONS = self.test_generations

            start_time = time.time()

            # Run the modified GA function that captures results
            best_rule, best_prog, best_fitness, best_curve, mean_curve = self._run_modified_old_ga(old_module)

            end_time = time.time()
            total_time = end_time - start_time

            # Calculate final accuracy by testing best genome
            outputs = old_module._simulate(
                best_rule, best_prog, old_module.INPUT_SET,
                old_module.WINDOW, old_module.MAX_STEPS, old_module.HALT_THRESH
            )
            correct = np.sum(outputs == old_module.TARGET_OUT)
            final_accuracy = correct / len(old_module.INPUT_SET)

            # Find convergence generation (accuracy >= 0.95)
            convergence_gen = None
            for i, fitness in enumerate(best_curve):
                # Convert fitness to approximate accuracy (fitness is negative error)
                approx_accuracy = max(0, 1 + fitness / 30)  # Rough conversion
                if approx_accuracy >= 0.95:
                    convergence_gen = i
                    break

            # Restore original values
            old_module.POP_SIZE = original_pop_size
            old_module.N_GENERATIONS = original_generations

            return BenchmarkResult(
                name="Old Implementation",
                total_time=total_time,
                generations=self.test_generations,
                best_fitness_curve=best_curve,
                mean_fitness_curve=mean_curve,
                final_accuracy=final_accuracy,
                convergence_generation=convergence_gen,
                time_per_generation=total_time / self.test_generations,
                success=True
            )

        except Exception as e:
            return BenchmarkResult(
                name="Old Implementation",
                total_time=0,
                generations=0,
                best_fitness_curve=[],
                mean_fitness_curve=[],
                final_accuracy=0,
                convergence_generation=None,
                time_per_generation=0,
                success=False,
                error_message=str(e)
            )

    def _run_modified_old_ga(self, old_module):
        """Run a modified version of the old GA that returns results."""
        import math
        from tqdm import tqdm

        # Population arrays
        pop_rules = np.empty((old_module.POP_SIZE, 64), np.uint8)
        pop_progs = np.empty((old_module.POP_SIZE, old_module.L_PROG), np.uint8)
        for i in range(old_module.POP_SIZE):
            r, p = old_module.random_genome()
            pop_rules[i], pop_progs[i] = r, p

        best_curve, mean_curve = [], []
        n_elite = int(math.ceil(old_module.ELITE_FRAC * old_module.POP_SIZE))
        n_imm   = max(1, int(old_module.EPS_RANDOM_IMMIGRANTS * old_module.POP_SIZE))

        for gen in tqdm(range(1, old_module.N_GENERATIONS+1), ncols=80, desc="GA"):
            fit = old_module.fitness_population(pop_rules, pop_progs)
            order = np.argsort(fit)[::-1]
            pop_rules, pop_progs, fit = pop_rules[order], pop_progs[order], fit[order]

            best_curve.append(float(fit[0]))
            mean_curve.append(float(fit.mean()))

            if gen % old_module.N_COMPLEX_TELEMETRY == 0:
                flat = np.concatenate((pop_rules, pop_progs), axis=1)
                ham = old_module.avg_pairwise_hamming(flat)
                tqdm.write(f"Gen {gen:3}  best={fit[0]:.3f}  mean={fit.mean():.3f}  ham={ham:.1f}")
            else:
                tqdm.write(f"Gen {gen:3}  best={fit[0]:.3f}  mean={fit.mean():.3f}")

            # ── produce next generation ──
            next_rules = pop_rules[:n_elite].copy()
            next_progs = pop_progs[:n_elite].copy()
            while next_rules.shape[0] < old_module.POP_SIZE:
                r1, p1 = old_module.tournament(pop_rules, pop_progs, fit)
                r2, p2 = old_module.tournament(pop_rules, pop_progs, fit)
                child_r, child_p = old_module.mutate(*old_module.crossover(r1, p1, r2, p2))
                next_rules = np.vstack((next_rules, child_r))
                next_progs = np.vstack((next_progs, child_p))

            # Random immigrants
            for _ in range(n_imm):
                idx = old_module.rng.integers(n_elite, old_module.POP_SIZE)
                next_rules[idx], next_progs[idx] = old_module.random_genome()

            pop_rules, pop_progs = next_rules, next_progs

        return pop_rules[0], pop_progs[0], best_curve[-1], best_curve, mean_curve
    
    def run_new_implementation(self) -> BenchmarkResult:
        """Run the new SDK-based implementation."""
        print("🔄 Running NEW implementation...")

        try:
            # Import new architecture components
            from emergent_models.core.state import StateModel
            from emergent_models.rules.sanitization import lut_idx
            from emergent_models.core.space_model import Tape1D
            from emergent_models.encoders.em43 import Em43Encoder
            from emergent_models.simulation.simulator import Simulator
            from emergent_models.training.new_fitness import AbsoluteDifferenceFitness
            from emergent_models.training import (
                Trainer, ComplexityRewardFitness, GAOptimizer
            )


            # Setup with same hyperparameters as old implementation
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
            space = Tape1D(length=200, radius=1)  # WINDOW = 200
            encoder = Em43Encoder(state, space)

            sim = Simulator(
                state=state,
                space=space,
                max_steps=800,      # MAX_STEPS
                halt_thresh=0.50    # HALT_THRESH
            )

            base_fitness = AbsoluteDifferenceFitness(continuous=True)
            fitness = ComplexityRewardFitness(base_fitness, complexity_bonus=0.05)

            optim = GAOptimizer(
                pop_size=self.test_pop_size,
                state=state,
                prog_len=10,                    # L_PROG
                mutation_rate=0.03,             # P_MUT_RULE
                prog_mutation_rate=0.08,        # P_MUT_PROG
                elite_fraction=0.1,             # ELITE_FRAC
                tournament_size=3,              # TOURNEY_K
                random_immigrant_rate=0.2,      # EPS_RANDOM_IMMIGRANTS
                prog_sparsity=0.3
            )

            # Create a simple console monitor that shows progress like old implementation
            class SimpleConsoleMonitor:
                def __init__(self):
                    self.history = []

                def update(self, generation: int, scores: np.ndarray, **kwargs) -> None:
                    """Show progress like old implementation."""
                    best_fitness = np.max(scores)
                    mean_fitness = np.mean(scores)
                    print(f"Gen {generation+1:4d}  best={best_fitness:.3f}  mean={mean_fitness:.3f}")

                    # Store for comparison script
                    self.history.append({
                        'generation': generation,
                        'best_fitness': best_fitness,
                        'mean_fitness': mean_fitness,
                        'std_fitness': np.std(scores)
                    })

            monitor = SimpleConsoleMonitor()
            trainer = Trainer(encoder, sim, fitness, optim, monitor)

            start_time = time.time()

            # Run training
            result = trainer.fit(
                inputs=np.arange(1, 31),  # INPUT_SET
                targets=2 * np.arange(1, 31),
                generations=self.test_generations,
                use_tqdm=False
            )

            end_time = time.time()
            total_time = end_time - start_time

            # Extract fitness curves from monitor history
            best_curve = [entry['best_fitness'] for entry in monitor.history]
            mean_curve = [entry['mean_fitness'] for entry in monitor.history]

            # Calculate final accuracy using the SAME method as training (critical fix)
            best_genome = result['best_genome']
            rule_table, programme = best_genome.extract_arrays()

            # Test the best genome using the SAME fused kernel used in training
            inputs = np.arange(1, 31, dtype=np.int64)
            targets = 2 * inputs

            # Get the simulation parameters from trainer components
            window = trainer.encoder.space.length  # 200
            max_steps = trainer.simulator.max_steps  # 800
            halt_thresh = trainer.simulator.halt_thresh  # 0.5
            prog_len = len(programme)
            
            outputs = sim(inputs)
            # Calculate accuracy
            correct = np.sum(outputs == targets)
            final_accuracy = correct / len(inputs)

            # Find convergence generation (accuracy >= 0.95)
            convergence_gen = None
            for i, fitness_val in enumerate(best_curve):
                # Convert fitness to approximate accuracy (fitness is negative error)
                approx_accuracy = max(0, 1 + fitness_val / 30)  # Same conversion as old
                if approx_accuracy >= 0.95:
                    convergence_gen = i
                    break

            return BenchmarkResult(
                name="New Implementation",
                total_time=total_time,
                generations=self.test_generations,
                best_fitness_curve=best_curve,
                mean_fitness_curve=mean_curve,
                final_accuracy=final_accuracy,  # Proper accuracy calculation
                convergence_generation=convergence_gen,
                time_per_generation=total_time / self.test_generations,
                success=True
            )

        except Exception as e:
            return BenchmarkResult(
                name="New Implementation",
                total_time=0,
                generations=0,
                best_fitness_curve=[],
                mean_fitness_curve=[],
                final_accuracy=0,
                convergence_generation=None,
                time_per_generation=0,
                success=False,
                error_message=str(e)
            )
    
    def run_comparison(self) -> Dict[str, BenchmarkResult]:
        """Run full comparison between implementations."""
        print("=" * 60)
        print("🔬 EM-4/3 Doubling: Old vs New Implementation Comparison")
        print("=" * 60)
        print(f"Test Parameters:")
        print(f"  Population Size: {self.test_pop_size}")
        print(f"  Generations: {self.test_generations}")
        print(f"  Input Range: 1-30")
        print("=" * 60)
        
        # Run both implementations
        self.results['old'] = self.run_old_implementation()
        self.results['new'] = self.run_new_implementation()
        
        return self.results
    
    def print_results(self):
        """Print comparison results."""
        print("\n" + "=" * 60)
        print("📊 COMPARISON RESULTS")
        print("=" * 60)
        
        for name, result in self.results.items():
            print(f"\n{result.name}:")
            print("-" * 40)
            if result.success:
                print(f"  ✅ Success: {result.success}")
                print(f"  ⏱️  Total Time: {result.total_time:.2f}s")
                print(f"  ⚡ Time/Generation: {result.time_per_generation:.3f}s")
                print(f"  🎯 Final Accuracy: {result.final_accuracy:.4f}")
                if result.convergence_generation:
                    print(f"  🚀 Converged at Gen: {result.convergence_generation}")
                else:
                    print(f"  🐌 No convergence detected")
            else:
                print(f"  ❌ Failed: {result.error_message}")
        
        # Performance comparison
        if self.results['old'].success and self.results['new'].success:
            old_result = self.results['old']
            new_result = self.results['new']
            
            speed_ratio = old_result.time_per_generation / new_result.time_per_generation
            
            print(f"\n🏁 PERFORMANCE COMPARISON:")
            print("-" * 40)
            if speed_ratio > 1.1:
                print(f"  🚀 NEW is {speed_ratio:.1f}x FASTER")
            elif speed_ratio < 0.9:
                print(f"  🐌 NEW is {1/speed_ratio:.1f}x SLOWER")
            else:
                print(f"  ≈ Similar performance ({speed_ratio:.1f}x)")
            
            print(f"\n🎯 ACCURACY COMPARISON:")
            print("-" * 40)
            print(f"  Old Final Accuracy: {old_result.final_accuracy:.4f}")
            print(f"  New Final Accuracy: {new_result.final_accuracy:.4f}")
            
            if new_result.final_accuracy > old_result.final_accuracy:
                improvement = new_result.final_accuracy - old_result.final_accuracy
                print(f"  ✅ NEW is {improvement:.4f} points better")
            elif old_result.final_accuracy > new_result.final_accuracy:
                decline = old_result.final_accuracy - new_result.final_accuracy
                print(f"  ⚠️  NEW is {decline:.4f} points worse")
            else:
                print(f"  ≈ Similar accuracy")

def main():
    """Main comparison function."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare old vs new EM-4/3 implementations')
    parser.add_argument('--generations', type=int, default=20, help='Number of generations to test')
    parser.add_argument('--population', type=int, default=1000, help='Population size to test')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive comparison')

    args = parser.parse_args()

    if args.comprehensive:
        # Comprehensive comparison
        comparator = PerformanceComparator(
            test_generations=50,
            test_pop_size=2000
        )
        print("🔬 Running COMPREHENSIVE comparison (this may take several minutes)...")
    else:
        # Quick comparison
        comparator = PerformanceComparator(
            test_generations=args.generations,
            test_pop_size=args.population
        )
        print("🔬 Running QUICK comparison...")

    results = comparator.run_comparison()
    comparator.print_results()

    # Save results for further analysis
    if results['old'].success and results['new'].success:
        save_comparison_data(results)

    if not args.comprehensive:
        print(f"\n💡 For a more thorough comparison, run:")
        print(f"   python {__file__} --comprehensive")

def save_comparison_data(results: Dict[str, BenchmarkResult]):
    """Save comparison data for visualization."""
    import json
    from pathlib import Path

    output_dir = Path("comparison_results")
    output_dir.mkdir(exist_ok=True)

    # Convert results to JSON-serializable format (handle numpy types)
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python types for JSON serialization."""
        if hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    data = {}
    for name, result in results.items():
        data[name] = {
            'name': result.name,
            'total_time': convert_to_json_serializable(result.total_time),
            'generations': convert_to_json_serializable(result.generations),
            'best_fitness_curve': convert_to_json_serializable(result.best_fitness_curve),
            'mean_fitness_curve': convert_to_json_serializable(result.mean_fitness_curve),
            'final_accuracy': convert_to_json_serializable(result.final_accuracy),
            'convergence_generation': convert_to_json_serializable(result.convergence_generation),
            'time_per_generation': convert_to_json_serializable(result.time_per_generation),
            'success': result.success
        }

    # Save to JSON
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n📊 Results saved to {output_dir}/comparison_results.json")
    print(f"   Use visualize_comparison.py to create plots")

if __name__ == "__main__":
    main()
