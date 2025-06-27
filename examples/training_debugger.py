#!/usr/bin/env python3
"""
Training Debugger
=================

Debug why training isn't working and provide solutions.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from emergent_models.core.state import StateModel
from emergent_models.core.space_model import Tape1D
from emergent_models.encoders.em43 import Em43Encoder
from emergent_models.simulation.simulator import Simulator
from emergent_models.training.new_fitness import AbsoluteDifferenceFitness
from emergent_models.training.optimizer import GAOptimizer
from emergent_models.training.new_trainer import Trainer
from emergent_models.core.base import ConsoleLogger


def analyze_population_diversity(population):
    """Analyze diversity in the population."""
    
    print("🔍 POPULATION DIVERSITY ANALYSIS")
    print("=" * 40)
    
    # Analyze programme diversity
    programmes = [genome.programme.code for genome in population]
    unique_programmes = len(set(tuple(p) for p in programmes))
    
    print(f"Population size: {len(population)}")
    print(f"Unique programmes: {unique_programmes}")
    print(f"Programme diversity: {unique_programmes/len(population):.2%}")
    
    # Analyze programme sparsity
    sparsities = [genome.programme.sparsity() for genome in population]
    print(f"Programme sparsity: {np.mean(sparsities):.3f} ± {np.std(sparsities):.3f}")
    
    # Analyze rule diversity
    rules = [genome.rule.table for genome in population]
    unique_rules = len(set(tuple(r) for r in rules))
    print(f"Unique rules: {unique_rules}")
    print(f"Rule diversity: {unique_rules/len(population):.2%}")
    
    # Show some example programmes
    print(f"\nExample programmes:")
    for i in range(min(5, len(population))):
        prog = population[i].programme.code
        sparsity = population[i].programme.sparsity()
        print(f"  {i+1}: {prog} (sparsity: {sparsity:.2f})")


def test_fitness_function():
    """Test if the fitness function works correctly."""
    
    print("\n🧪 FITNESS FUNCTION TEST")
    print("=" * 30)
    
    fitness = AbsoluteDifferenceFitness()
    
    # Test perfect outputs
    inputs = np.array([1, 2, 3, 4, 5])
    perfect_outputs = 2 * inputs
    perfect_scores = fitness(perfect_outputs, inputs)
    
    print(f"Perfect case:")
    print(f"  Inputs: {inputs}")
    print(f"  Outputs: {perfect_outputs}")
    print(f"  Scores: {perfect_scores}")
    print(f"  Mean score: {np.mean(perfect_scores):.3f}")
    
    # Test random outputs
    random_outputs = np.random.randint(0, 20, len(inputs))
    random_scores = fitness(random_outputs, inputs)
    
    print(f"\nRandom case:")
    print(f"  Inputs: {inputs}")
    print(f"  Outputs: {random_outputs}")
    print(f"  Scores: {random_scores}")
    print(f"  Mean score: {np.mean(random_scores):.3f}")


def analyze_training_progress(trainer, inputs, generations=10):
    """Analyze training progress step by step."""
    
    print(f"\n📈 TRAINING PROGRESS ANALYSIS")
    print("=" * 40)
    
    print(f"Training for {generations} generations...")
    
    history = {
        'best_fitness': [],
        'mean_fitness': [],
        'diversity': []
    }
    
    for gen in range(generations):
        # Get population
        population = trainer.optimizer.ask()
        
        # Analyze diversity
        programmes = [genome.programme.code for genome in population]
        unique_programmes = len(set(tuple(p) for p in programmes))
        diversity = unique_programmes / len(population)
        
        # Evaluate
        scores = trainer.evaluate_population(population, inputs)
        
        # Update optimizer
        trainer.optimizer.tell(scores)
        
        # Record stats
        best_score = np.max(scores)
        mean_score = np.mean(scores)
        
        history['best_fitness'].append(best_score)
        history['mean_fitness'].append(mean_score)
        history['diversity'].append(diversity)
        
        print(f"Gen {gen:2d}: Best={best_score:.4f}, Mean={mean_score:.4f}, Diversity={diversity:.2%}")
        
        # Analyze best genome
        if gen % 5 == 0:
            best_idx = np.argmax(scores)
            best_genome = population[best_idx]
            print(f"        Best programme: {best_genome.programme.code}")
    
    return history


def suggest_improvements():
    """Suggest improvements to make training work."""
    
    print(f"\n💡 IMPROVEMENT SUGGESTIONS")
    print("=" * 40)
    
    print("Why training isn't working:")
    print("1. 🎯 Random initialization: Rules start completely random")
    print("2. 🔍 Search space: 4^64 possible rules (huge!)")
    print("3. 🧬 Complexity: Doubling requires sophisticated rule interactions")
    print("4. 🎲 Fitness landscape: Very sparse rewards")
    
    print(f"\nSolutions to try:")
    print("1. 🌱 Better initialization:")
    print("   - Start with rules that have basic propagation")
    print("   - Use domain knowledge to seed population")
    
    print("2. 📊 Improved fitness:")
    print("   - Reward partial progress (beacon movement)")
    print("   - Use continuous fitness instead of binary")
    print("   - Multi-objective: correctness + simplicity")
    
    print("3. 🔧 Algorithm improvements:")
    print("   - Larger population size")
    print("   - Lower mutation rates")
    print("   - Elitism to preserve good solutions")
    
    print("4. 🎯 Simpler tasks first:")
    print("   - Start with identity function (f(x) = x)")
    print("   - Then increment (f(x) = x + 1)")
    print("   - Finally doubling (f(x) = 2x)")


def create_better_trainer():
    """Create a trainer with better settings for debugging."""
    
    print(f"\n🔧 CREATING IMPROVED TRAINER")
    print("=" * 40)
    
    # Setup with better parameters
    state = StateModel([0, 1, 2, 3], immutable={0: 0})
    space = Tape1D(length=100, radius=1)  # Smaller for faster debugging
    encoder = Em43Encoder(state, space)
    sim = Simulator(state, space, max_steps=50, halt_thresh=0.5)
    
    # Use continuous fitness
    fitness = AbsoluteDifferenceFitness(continuous=True)
    
    # Better optimizer settings
    optim = GAOptimizer(
        pop_size=50,  # Smaller for debugging
        state=state,
        prog_len=8,   # Shorter programmes
        mutation_rate=0.01,  # Lower mutation
        prog_mutation_rate=0.05,
        elite_fraction=0.2,  # More elitism
        random_immigrant_rate=0.1  # Less random immigrants
    )
    
    monitor = ConsoleLogger(log_every=2)
    trainer = Trainer(encoder, sim, fitness, optim, monitor)
    
    print("Improved settings:")
    print(f"  Population: {optim.pop_size}")
    print(f"  Programme length: {optim.prog_len}")
    print(f"  Mutation rate: {optim.mutation_rate}")
    print(f"  Elite fraction: {optim.elite_fraction}")
    print(f"  Continuous fitness: True")
    
    return trainer


def main():
    """Main debugging function."""
    
    print("🔬 Training Debugger")
    print("=" * 40)
    
    # 1. Test fitness function
    test_fitness_function()
    
    # 2. Create improved trainer
    trainer = create_better_trainer()
    
    # 3. Analyze initial population
    population = trainer.optimizer.ask()
    analyze_population_diversity(population)
    
    # 4. Run short training
    inputs = np.array([1, 2, 3])  # Smaller input set for debugging
    history = analyze_training_progress(trainer, inputs, generations=10)
    
    # 5. Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['best_fitness'], 'b-', label='Best')
    plt.plot(history['mean_fitness'], 'r-', label='Mean')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['diversity'], 'g-')
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.title('Population Diversity')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.scatter(history['diversity'], history['best_fitness'], alpha=0.7)
    plt.xlabel('Diversity')
    plt.ylabel('Best Fitness')
    plt.title('Diversity vs Fitness')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 6. Suggestions
    suggest_improvements()
    
    print(f"\n✅ Debugging complete!")
    print("Next steps:")
    print("1. Try the improved trainer settings")
    print("2. Implement better initialization")
    print("3. Use continuous fitness")
    print("4. Start with simpler tasks")


if __name__ == "__main__":
    main()
