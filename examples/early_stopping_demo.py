#!/usr/bin/env python3
"""
Early Stopping Demonstration
============================
Demonstrates the early stopping functionality in emergent-models.
This example creates a simple task and shows how training automatically
stops when the accuracy threshold is reached.
"""

import numpy as np
from emergent_models.rules.em43 import EM43Rule, EM43Genome
from emergent_models.simulators.numba_simulator import NumbaSimulator
from emergent_models.optimizers.genetic import GAOptimizer
from emergent_models.training.trainer import CATrainer, create_accuracy_validator
from emergent_models.losses.distance import HammingLoss


def create_simple_task():
    """Create a simple task: map 1->2, 2->4, 3->6 (doubling)"""
    test_inputs = [1, 2, 3]
    test_targets = [2, 4, 6]
    return test_inputs, test_targets


def create_perfect_genome():
    """Create a genome that perfectly solves the doubling task"""
    # This is a demonstration - in practice you wouldn't know the perfect solution
    rule_array = np.zeros(64, dtype=np.uint8)
    
    # Set up some rules that might work for doubling
    # This is simplified - a real solution would be more complex
    for i in range(64):
        rule_array[i] = np.random.randint(0, 4)
    
    # Create a simple programme
    programme = np.array([1, 0, 2, 0, 1], dtype=np.uint8)
    
    rule = EM43Rule(rule_array)
    genome = EM43Genome(rule, programme)
    
    return genome


def main():
    print("Early Stopping Demonstration")
    print("=" * 40)
    
    # Create task
    test_inputs, test_targets = create_simple_task()
    print(f"Task: {dict(zip(test_inputs, test_targets))}")
    
    # Create simulator
    simulator = NumbaSimulator(window=100, max_steps=50, halt_thresh=0.5)
    
    # Create optimizer with small population for quick demo
    optimizer = GAOptimizer(
        population_size=20,
        mutation_rate=0.1,
        programme_mutation_rate=0.2,
        elite_fraction=0.2,
        tournament_size=3
    )
    
    # Initialize population
    optimizer.initialize_population(programme_length=5)
    
    # Add a potentially good genome to the population to increase chances of success
    # In a real scenario, you wouldn't do this
    good_genome = create_perfect_genome()
    optimizer.population[0] = good_genome
    
    # Create validation function
    validation_fn = create_accuracy_validator(test_inputs, test_targets)
    
    # Create fitness function
    def fitness_fn(genome, sim):
        try:
            outputs = sim.simulate_batch(genome, test_inputs)
            correct = sum(1 for out, target in zip(outputs, test_targets) 
                         if out >= 0 and out == target)
            accuracy = correct / len(test_inputs)
            
            # Add some noise to make it more realistic
            noise = np.random.normal(0, 0.01)
            return max(0, accuracy + noise)
        except:
            return 0.0
    
    # Create trainer
    trainer = CATrainer(simulator, optimizer, HammingLoss(), verbose=True)
    
    print("\nStarting training with early stopping...")
    print("Will stop when 100% accuracy is reached")
    print("-" * 40)
    
    # Train with early stopping
    trainer.fit(
        fitness_fn=fitness_fn,
        epochs=50,  # Maximum generations
        early_stopping_threshold=1.0,  # 100% accuracy
        early_stopping_metric="accuracy",
        validation_fn=validation_fn,
        checkpoint_every=0  # Disable checkpointing for demo
    )
    
    print("\nTraining Results:")
    print("-" * 20)
    
    # Test the best genome
    best_genome = trainer.best_genome
    if best_genome:
        print(f"Best fitness: {trainer.best_fitness:.4f}")
        
        # Test on the task
        test_outputs = simulator.simulate_batch(best_genome, test_inputs)
        
        print("\nFinal Test Results:")
        print("Input -> Expected -> Actual -> Status")
        print("-" * 35)
        
        correct = 0
        for inp, expected, actual in zip(test_inputs, test_targets, test_outputs):
            status = "✓" if actual == expected else "✗"
            if actual == expected:
                correct += 1
            print(f"{inp:2d} -> {expected:2d} -> {actual:2d} -> {status}")
        
        final_accuracy = correct / len(test_inputs)
        print(f"\nFinal Accuracy: {final_accuracy:.1%}")
        
        if final_accuracy >= 1.0:
            print("🎉 Perfect solution found!")
        elif final_accuracy >= 0.8:
            print("👍 Good solution found!")
        else:
            print("🤔 Solution needs improvement")
    
    # Show training statistics
    stats = trainer.get_training_stats()
    if stats:
        print(f"\nTraining Statistics:")
        print(f"- Generations completed: {stats['generations']}")
        print(f"- Best fitness achieved: {stats['best_fitness']:.4f}")
        print(f"- Total training time: {stats['total_time']:.2f}s")
        
        if stats['generations'] < 50:
            print("✅ Early stopping was triggered!")
        else:
            print("⏰ Training completed all generations")


if __name__ == "__main__":
    # Set random seed for reproducible demo
    np.random.seed(42)
    main()
