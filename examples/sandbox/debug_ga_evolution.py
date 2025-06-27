#!/usr/bin/env python3
"""
Debug GA Evolution Issues

This script investigates why fitness values remain constant across generations
in both old and new implementations.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def debug_old_implementation():
    """Debug the old implementation GA evolution."""
    print("🔍 Debugging OLD Implementation GA Evolution")
    print("=" * 50)
    
    # Import old implementation
    import examples.sandbox.em43_doubling_old as old_module
    
    # Test with small population for debugging
    old_module.POP_SIZE = 10
    old_module.N_GENERATIONS = 3
    
    print(f"Population size: {old_module.POP_SIZE}")
    print(f"Generations: {old_module.N_GENERATIONS}")
    
    # Initialize population
    pop_rules = np.empty((old_module.POP_SIZE, 64), np.uint8)
    pop_progs = np.empty((old_module.POP_SIZE, old_module.L_PROG), np.uint8)
    for i in range(old_module.POP_SIZE):
        r, p = old_module.random_genome()
        pop_rules[i], pop_progs[i] = r, p
    
    print(f"\n📊 Initial Population Analysis:")
    print(f"Rule shapes: {pop_rules.shape}")
    print(f"Programme shapes: {pop_progs.shape}")
    print(f"Sample rule: {pop_rules[0][:10]}...")
    print(f"Sample programme: {pop_progs[0]}")
    
    # Test fitness calculation
    print(f"\n🧮 Testing Fitness Calculation...")
    fit = old_module.fitness_population(pop_rules, pop_progs)
    print(f"Fitness shape: {fit.shape}")
    print(f"Fitness values: {fit}")
    print(f"Fitness range: [{fit.min():.3f}, {fit.max():.3f}]")
    print(f"Fitness std: {fit.std():.3f}")
    
    # Test if fitness values are all identical
    if fit.std() < 1e-6:
        print("⚠️  WARNING: All fitness values are nearly identical!")
        print("This suggests a problem with fitness calculation or simulation.")
    
    # Test simulation on a single genome
    print(f"\n🔬 Testing Single Genome Simulation...")
    test_rule = pop_rules[0]
    test_prog = pop_progs[0]
    test_inputs = np.array([1, 2, 3], dtype=np.int64)
    
    outputs = old_module._simulate(test_rule, test_prog, test_inputs, 
                                 old_module.WINDOW, old_module.MAX_STEPS, old_module.HALT_THRESH)
    print(f"Test inputs: {test_inputs}")
    print(f"Test outputs: {outputs}")
    print(f"Expected outputs: {2 * test_inputs}")
    print(f"Errors: {np.abs(outputs - 2 * test_inputs)}")
    
    # Test GA evolution for a few generations
    print(f"\n🧬 Testing GA Evolution...")
    n_elite = int(np.ceil(old_module.ELITE_FRAC * old_module.POP_SIZE))
    n_imm = max(1, int(old_module.EPS_RANDOM_IMMIGRANTS * old_module.POP_SIZE))
    
    for gen in range(1, 4):
        print(f"\n--- Generation {gen} ---")
        
        # Evaluate fitness
        fit = old_module.fitness_population(pop_rules, pop_progs)
        order = np.argsort(fit)[::-1]
        pop_rules, pop_progs, fit = pop_rules[order], pop_progs[order], fit[order]
        
        print(f"Best fitness: {fit[0]:.6f}")
        print(f"Mean fitness: {fit.mean():.6f}")
        print(f"Fitness std: {fit.std():.6f}")
        
        # Check if population is evolving
        if gen > 1:
            if abs(fit[0] - prev_best) < 1e-8:
                print("⚠️  WARNING: Best fitness not improving!")
            if abs(fit.mean() - prev_mean) < 1e-8:
                print("⚠️  WARNING: Mean fitness not changing!")
        
        prev_best = fit[0]
        prev_mean = fit.mean()
        
        # Produce next generation
        next_rules = pop_rules[:n_elite].copy()
        next_progs = pop_progs[:n_elite].copy()
        
        print(f"Elite size: {n_elite}")
        print(f"Immigrants: {n_imm}")
        
        # Test tournament selection
        r1, p1 = old_module.tournament(pop_rules, pop_progs, fit)
        r2, p2 = old_module.tournament(pop_rules, pop_progs, fit)
        print(f"Selected parents have different rules: {not np.array_equal(r1, r2)}")
        print(f"Selected parents have different progs: {not np.array_equal(p1, p2)}")
        
        # Test crossover
        child_r, child_p = old_module.crossover(r1, p1, r2, p2)
        print(f"Child differs from parent1 rule: {not np.array_equal(child_r, r1)}")
        print(f"Child differs from parent1 prog: {not np.array_equal(child_p, p1)}")
        
        # Test mutation
        mut_r, mut_p = old_module.mutate(child_r, child_p)
        print(f"Mutation changed rule: {not np.array_equal(mut_r, child_r)}")
        print(f"Mutation changed prog: {not np.array_equal(mut_p, child_p)}")
        
        # Generate full next generation
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
        
        # Check population diversity
        flat = np.concatenate((pop_rules, pop_progs), axis=1)
        unique_genomes = len(np.unique(flat.view(np.void), axis=0))
        print(f"Unique genomes: {unique_genomes}/{old_module.POP_SIZE}")


def debug_new_implementation():
    """Debug the new implementation GA evolution."""
    print("\n🔍 Debugging NEW Implementation GA Evolution")
    print("=" * 50)
    
    from emergent_models.core.state import StateModel
    from emergent_models.rules.sanitization import lut_idx
    from emergent_models.core.space_model import Tape1D
    from emergent_models.encoders.em43 import Em43Encoder
    from emergent_models.simulation.simulator import Simulator
    from emergent_models.training.new_fitness import AbsoluteDifferenceFitness, ComplexityRewardFitness
    from emergent_models.training.optimizer import GAOptimizer
    from emergent_models.training.new_trainer import Trainer
    from emergent_models.training.monitor import SilentMonitor
    
    # Setup (small for debugging)
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
    
    sim = Simulator(state=state, space=space, max_steps=800, halt_thresh=0.50)
    
    base_fitness = AbsoluteDifferenceFitness(continuous=True)
    fitness = ComplexityRewardFitness(base_fitness, complexity_bonus=0.05)
    
    optim = GAOptimizer(
        pop_size=10,  # Small for debugging
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
    
    print(f"Population size: {optim.pop_size}")
    
    # Test initial population
    population = optim.ask()
    print(f"Initial population size: {len(population)}")
    
    # Test fitness evaluation
    inputs = np.array([1, 2, 3], dtype=np.int64)
    fitness_scores = trainer.evaluate_population(population, inputs)
    print(f"Fitness scores: {fitness_scores}")
    print(f"Fitness range: [{fitness_scores.min():.3f}, {fitness_scores.max():.3f}]")
    print(f"Fitness std: {fitness_scores.std():.3f}")
    
    # Test evolution
    optim.tell(fitness_scores)
    next_population = optim.ask()
    
    print(f"Population evolved successfully: {len(next_population) == len(population)}")


if __name__ == "__main__":
    try:
        debug_old_implementation()
        debug_new_implementation()
        
    except Exception as e:
        print(f"\n❌ Debug failed with error: {e}")
        import traceback
        traceback.print_exc()
