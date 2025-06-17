# Emergent Models

A PyTorch-like library for training cellular automata and emergent computational models.


## Overview

Emergent Models provides a familiar PyTorch-like interface for experimenting with cellular automata and other emergent computational systems. It supports:

- **Multiple Space Types**: 1D tapes, 2D grids, and custom topologies
- **Flexible Rule Systems**: Elementary CA, totalistic rules, neural CA, and custom rules
- **Genetic Optimization**: Built-in genetic algorithms for evolving CA rules and programs
- **Rich Ecosystem**: Datasets, transforms, visualization tools, and pre-trained models

## Installation

```bash
pip install emergent-models
```

For development:
```bash
git clone https://github.com/yourusername/emergent-models.git
cd emergent-models
pip install -e ".[dev]"
```

## Quick Start

```python
import emergent_models as em

# Create a 1D elementary cellular automaton
ca = em.ElementaryCA(rule_number=110)

# Initialize a random binary space
space = em.Space1D(size=100, n_states=2)
space.randomize()

# Create a simulator
simulator = em.Simulator(max_steps=50)

# Run the simulation
final_state = simulator(ca, space)

# Visualize the evolution
em.visualize_evolution(simulator.history)
```

## Training Example

```python
import emergent_models as em
from emergent_models.data import BinaryAdditionDataset
from emergent_models.training import CATrainer

# Create dataset
dataset = BinaryAdditionDataset(n_samples=1000)
dataloader = em.CADataLoader(dataset, batch_size=32)

# Initialize population of genomes
population = [em.Genome(em.ElementaryCA(rule)) for rule in range(256)]

# Setup training
simulator = em.Simulator(max_steps=100)
optimizer = em.GAOptimizer(population_size=256, mutation_rate=0.02)
loss_fn = em.HammingLoss()

trainer = CATrainer(simulator, optimizer, loss_fn)

# Train
trainer.fit(population, dataloader, epochs=50)

# Get best genome
best_genome = trainer.best_genome
```

## Key Features

### 1. PyTorch-like API

```python
# Modules with forward() method
class MyCA(em.CAModule):
    def forward(self, x):
        return self.rule(x)

# Composable transforms
transform = em.Compose([
    em.BinaryEncoder(),
    em.PadSpace(left=10, right=10)
])

```

### 2. Rich Rule Library

```python
# Elementary CA
rule110 = em.ElementaryCA(110)

# Totalistic CA
life = em.TotalisticCA2D(birth=[3], survival=[2, 3])

# Neural CA
neural_ca = em.NeuralCA(hidden_dim=64, n_states=16)

# Custom rules
@em.rule_function
def my_rule(neighborhood):
    return sum(neighborhood) % 3
```

### 3. Advanced Optimizers

```python
# Genetic Algorithm
ga = em.GAOptimizer(
    population_size=100,
    mutation_rate=0.01,
    crossover_rate=0.7,
    tournament_size=5
)

# Coevolution
coevo = em.CoevolutionOptimizer(
    n_species=5,
    migration_rate=0.1
)

# Evolution Strategies
es = em.EvolutionStrategy(
    population_size=50,
    sigma=0.1
)
```

### 4. Visualization Tools

```python
# Static visualization
em.plot_space(space, cmap='viridis')

# Evolution animation
em.animate_evolution(history, fps=10, filename='evolution.gif')

# Interactive visualization
em.interactive_ca(rule=110, size=200)
```


## Examples

The `examples/` directory contains:

- ...

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

```bash
# Run tests
make test

# Format code
make format

# Build docs
make docs
```

## Citation

If you use Emergent Models in your research, please cite:

```bibtex
@software{emergent_models,
  title = {Emergent Models: A PyTorch-like Library for Cellular Automata},
  author = {Umut Eser},
  year = {2025},
  url = {https://github.com/weavebio/emergent-models}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- https://github.com/BrightStarLabs/EM43 (inspired by)
- Inspired by PyTorch's design philosophy
- Built on the foundations of cellular automata theory
- Community contributions and feedback

## Roadmap

- [ ] Add more examples
