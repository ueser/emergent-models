from __future__ import annotations

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any

from ..core.genome import Genome
from ..rules.em43 import EM43Genome, EM43Rule


def save_genome(genome: Union[Genome, EM43Genome], filepath: str) -> None:
    """Save genome to file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(genome, EM43Genome):
        save_em43_genome(genome, filepath)
    else:
        # Generic genome saving using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(genome, f)


def load_genome(filepath: str) -> Union[Genome, EM43Genome]:
    """Load genome from file."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Genome file not found: {filepath}")

    # Try to load as EM43Genome first (JSON format)
    if filepath.suffix == '.json':
        return load_em43_genome(filepath)

    # Try pickle format
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception:
        # Fallback to EM43 format
        return load_em43_genome(filepath)


def save_em43_genome(genome: EM43Genome, filepath: Union[str, Path]) -> None:
    """Save EM43Genome in a human-readable JSON format."""
    filepath = Path(filepath)

    # Ensure .json extension
    if filepath.suffix != '.json':
        filepath = filepath.with_suffix('.json')

    data = {
        'type': 'EM43Genome',
        'rule_array': genome.rule.get_rule_array().tolist(),
        'programme': genome.programme.tolist(),
        'fitness': float(genome.fitness)
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_em43_genome(filepath: Union[str, Path]) -> EM43Genome:
    """Load EM43Genome from JSON format."""
    filepath = Path(filepath)

    with open(filepath, 'r') as f:
        data = json.load(f)

    if data.get('type') != 'EM43Genome':
        raise ValueError(f"Invalid genome type: {data.get('type')}")

    rule_array = np.array(data['rule_array'], dtype=np.uint8)
    programme = np.array(data['programme'], dtype=np.uint8)

    rule = EM43Rule(rule_array)
    genome = EM43Genome(rule, programme)
    genome.fitness = data.get('fitness', 0.0)

    return genome


def save_population(
    population: list,
    directory: Union[str, Path],
    prefix: str = "genome"
) -> None:
    """Save an entire population to a directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    for i, genome in enumerate(population):
        filename = f"{prefix}_{i:04d}.json"
        save_genome(genome, directory / filename)

    # Save population metadata
    metadata = {
        'population_size': len(population),
        'genome_type': type(population[0]).__name__ if population else 'Unknown'
    }

    with open(directory / 'population_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def load_population(directory: Union[str, Path]) -> list:
    """Load an entire population from a directory."""
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Population directory not found: {directory}")

    # Load all genome files
    genome_files = sorted(directory.glob("genome_*.json"))

    if not genome_files:
        raise ValueError(f"No genome files found in {directory}")

    population = []
    for genome_file in genome_files:
        genome = load_genome(genome_file)
        population.append(genome)

    return population


def save_training_checkpoint(
    trainer,
    filepath: Union[str, Path],
    include_population: bool = True
) -> None:
    """Save a complete training checkpoint."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        'generation': trainer.optimizer.generation,
        'best_fitness': trainer.best_fitness,
        'history': trainer.history,
        'optimizer_config': {
            'population_size': trainer.optimizer.population_size,
            'mutation_rate': trainer.optimizer.mutation_rate,
            'elite_fraction': trainer.optimizer.elite_fraction,
            'tournament_size': trainer.optimizer.tournament_size
        }
    }

    # Save best genome separately
    if trainer.best_genome is not None:
        best_genome_path = filepath.with_suffix('.best_genome.json')
        save_genome(trainer.best_genome, best_genome_path)
        checkpoint_data['best_genome_file'] = str(best_genome_path)

    # Save population if requested
    if include_population and trainer.optimizer.population:
        population_dir = filepath.parent / f"{filepath.stem}_population"
        save_population(trainer.optimizer.population, population_dir)
        checkpoint_data['population_directory'] = str(population_dir)

    # Save checkpoint metadata
    with open(filepath, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def load_training_checkpoint(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load a training checkpoint."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    with open(filepath, 'r') as f:
        checkpoint_data = json.load(f)

    # Load best genome if available
    if 'best_genome_file' in checkpoint_data:
        best_genome_path = Path(checkpoint_data['best_genome_file'])
        if best_genome_path.exists():
            checkpoint_data['best_genome'] = load_genome(best_genome_path)

    # Load population if available
    if 'population_directory' in checkpoint_data:
        population_dir = Path(checkpoint_data['population_directory'])
        if population_dir.exists():
            checkpoint_data['population'] = load_population(population_dir)

    return checkpoint_data
