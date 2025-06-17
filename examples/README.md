# Emergent Models Examples

This directory contains examples demonstrating how to use the emergent-models library for training cellular automata.

## Examples

### 1. EM-4/3 Doubling (`em43_doubling.py`)

Trains a 4-state cellular automaton to perform the doubling operation (x → 2x) using the EM-4/3 system.

**Features:**
- EM-4/3 cellular automaton with 4 states (empty, active, red beacon, blue separator)
- Numba-accelerated batch simulation
- Genetic algorithm optimization with tournament selection
- Automatic checkpointing and result visualization

**Usage:**
```bash
# Run training
python examples/em43_doubling.py

# Test a previously saved genome
python examples/em43_doubling.py test
```

**Expected Output:**
The training will evolve a population of CA genomes over multiple generations, showing:
- Best and mean fitness progression
- Final test accuracy on the doubling task
- Saved best genome and fitness curve

### 2. Increment (`increment.py`)

A simpler example that trains a CA to increment numbers (x → x+1).

**Usage:**
```bash
python examples/increment.py
```

### 3. Early Stopping Demo (`early_stopping_demo.py`)

Demonstrates the early stopping functionality that automatically terminates training when a target accuracy is reached.

**Features:**
- Automatic termination when 100% accuracy is achieved
- Validation-based early stopping
- Small population for quick demonstration

**Usage:**
```bash
python examples/early_stopping_demo.py
```

## Key Concepts Demonstrated

### EM-4/3 System
The EM-4/3 (Emergent Models 4-state, radius-3) system uses:
- **4 states**: 0 (empty), 1 (active), 2 (red beacon), 3 (blue separator)
- **Radius-1 neighborhood**: Each cell looks at its left and right neighbors
- **Programme + Input encoding**: The tape contains a programme, separator (BB), and input encoding
- **Halting condition**: Simulation stops when enough blue cells are present

### Genetic Algorithm
The GA optimizer includes:
- **Tournament selection**: Best individuals from random tournaments become parents
- **Crossover**: Single-point crossover for both rules and programmes
- **Mutation**: Random changes to rule entries and programme cells
- **Elitism**: Best individuals are preserved across generations
- **Random immigrants**: New random genomes are introduced to maintain diversity

### Numba Acceleration
The simulator uses Numba JIT compilation for:
- **Batch processing**: Evaluate multiple inputs simultaneously
- **Parallel execution**: Utilize multiple CPU cores
- **Optimized loops**: Fast CA rule application

## Early Stopping

The library supports automatic early stopping when a target performance is reached:

```python
# Create validation function
test_inputs = [1, 2, 3, 4, 5]
test_targets = [2, 4, 6, 8, 10]  # Doubling
validation_fn = create_accuracy_validator(test_inputs, test_targets)

# Train with early stopping
trainer.fit(
    fitness_fn=fitness_fn,
    epochs=200,
    early_stopping_threshold=1.0,  # Stop at 100% accuracy
    early_stopping_metric="accuracy",
    validation_fn=validation_fn
)
```

**Early stopping options:**
- `early_stopping_threshold`: Target value to reach (e.g., 1.0 for 100%)
- `early_stopping_metric`: What to monitor ("fitness", "accuracy", "validation")
- `validation_fn`: Function to compute validation accuracy

## Configuration

Key parameters you can adjust:

```python
config = {
    'population_size': 1000,      # Number of genomes in population
    'generations': 200,           # Number of evolution generations
    'programme_length': 10,       # Length of the CA programme
    'window_size': 200,          # Tape length for simulation
    'max_steps': 256,            # Maximum simulation steps
    'halt_thresh': 0.50,         # Halting threshold (fraction of blue cells)
    'mutation_rate': 0.03,       # Rule mutation rate
    'programme_mutation_rate': 0.08,  # Programme mutation rate
    'elite_fraction': 0.1,       # Fraction of population kept as elite
    'tournament_size': 3,        # Tournament selection size
    'random_immigrant_rate': 0.2  # Fraction of random immigrants per generation
}
```

## Results

Successful training typically achieves:
- **High accuracy**: 90%+ on the target mathematical operation
- **Convergence**: Fitness improvement over 100-200 generations
- **Generalization**: Works on test inputs not seen during training

The trained genomes are saved in JSON format and can be loaded for further analysis or deployment.

## Extending the Examples

To create your own CA training experiments:

1. **Define a new dataset** in `emergent_models/data/mathematical.py`
2. **Create a fitness function** that evaluates genome performance
3. **Adjust hyperparameters** for your specific problem
4. **Use the visualization tools** to analyze results

Example for a new mathematical operation:

```python
from emergent_models.data.mathematical import CADataset

class SquaringDataset(CADataset):
    def __init__(self, input_range=(1, 10)):
        self.inputs = list(range(input_range[0], input_range[1] + 1))
        self.targets = [x * x for x in self.inputs]
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        # Return input and target spaces
        # ... implementation details
```

## Troubleshooting

**Common issues:**

1. **Numba compilation errors**: Ensure numba is properly installed
2. **Memory issues**: Reduce population_size or window_size
3. **Slow convergence**: Increase mutation rates or population diversity
4. **No improvement**: Check fitness function and ensure it's properly rewarding correct behavior

**Performance tips:**

- Use Numba simulator for large populations
- Adjust batch sizes based on available memory
- Monitor generation times and adjust parameters accordingly
- Use checkpointing for long training runs
