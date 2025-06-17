from __future__ import annotations

from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ..core.space import CASpace, Space1D, Space2D


def visualize_evolution(
    history: List[CASpace],
    figsize: tuple = (12, 8),
    cmap: str = "viridis",
    save_path: Optional[str] = None
) -> None:
    """Visualize the evolution of a cellular automaton."""
    if not history:
        print("No history to visualize")
        return

    n_steps = len(history)

    # Handle 1D CA
    if isinstance(history[0], Space1D):
        visualize_1d_evolution(history, figsize, cmap, save_path)
    # Handle 2D CA
    elif isinstance(history[0], Space2D):
        visualize_2d_evolution(history, figsize, cmap, save_path)
    else:
        # Generic visualization
        fig, axes = plt.subplots(1, min(n_steps, 5), figsize=figsize)
        if n_steps == 1:
            axes = [axes]

        for i, (ax, space) in enumerate(zip(axes, history[:5])):
            im = ax.imshow(space.data, cmap=cmap, aspect='auto')
            ax.set_title(f"Step {i}")
            ax.set_xlabel("Position")
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def visualize_1d_evolution(
    history: List[Space1D],
    figsize: tuple = (12, 8),
    cmap: str = "viridis",
    save_path: Optional[str] = None
) -> None:
    """Visualize 1D CA evolution as a space-time diagram."""
    if not history:
        return

    # Create space-time matrix
    n_steps = len(history)
    size = history[0].size
    spacetime = np.zeros((n_steps, size))

    for t, space in enumerate(history):
        spacetime[t, :] = space.data

    plt.figure(figsize=figsize)
    plt.imshow(spacetime, cmap=cmap, aspect='auto', origin='upper')
    plt.xlabel("Space")
    plt.ylabel("Time")
    plt.title("Cellular Automaton Evolution")
    plt.colorbar(label="Cell State")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_2d_evolution(
    history: List[Space2D],
    figsize: tuple = (15, 5),
    cmap: str = "viridis",
    save_path: Optional[str] = None
) -> None:
    """Visualize 2D CA evolution as a series of snapshots."""
    if not history:
        return

    n_steps = len(history)
    n_cols = min(5, n_steps)

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i < n_steps:
            im = ax.imshow(history[i].data, cmap=cmap)
            ax.set_title(f"Step {i}")
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_fitness_curve(
    history: Dict[str, List[float]],
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """Plot fitness evolution over generations."""
    plt.figure(figsize=figsize)

    if 'best_fitness' in history:
        plt.plot(history['best_fitness'], label='Best Fitness', linewidth=2)

    if 'mean_fitness' in history:
        plt.plot(history['mean_fitness'], label='Mean Fitness', linewidth=2)

    if 'std_fitness' in history and 'mean_fitness' in history:
        mean = np.array(history['mean_fitness'])
        std = np.array(history['std_fitness'])
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3, label='±1 Std')

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_population_diversity(
    population: List[Any],
    metric_fn: callable = None,
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None
) -> None:
    """Plot population diversity metrics."""
    if not population:
        print("Empty population")
        return

    if metric_fn is None:
        # Default: Hamming distance for EM43 genomes
        def default_metric(pop):
            if hasattr(pop[0], 'rule') and hasattr(pop[0].rule, 'get_rule_array'):
                # EM43 genomes
                arrays = [genome.rule.get_rule_array() for genome in pop]
                distances = []
                for i in range(len(arrays)):
                    for j in range(i + 1, len(arrays)):
                        dist = np.sum(arrays[i] != arrays[j])
                        distances.append(dist)
                return distances
            else:
                return [0]  # Fallback

        metric_fn = default_metric

    distances = metric_fn(population)

    plt.figure(figsize=figsize)
    plt.hist(distances, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel("Pairwise Distance")
    plt.ylabel("Frequency")
    plt.title("Population Diversity")
    plt.grid(True, alpha=0.3)

    # Add statistics
    mean_dist = np.mean(distances)
    plt.axvline(mean_dist, color='red', linestyle='--', label=f'Mean: {mean_dist:.2f}')
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def animate_evolution(
    history: List[CASpace],
    interval: int = 200,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6)
) -> FuncAnimation:
    """Create an animation of CA evolution."""
    if not history:
        raise ValueError("No history to animate")

    fig, ax = plt.subplots(figsize=figsize)

    # Initialize with first frame
    if isinstance(history[0], Space1D):
        line, = ax.plot(history[0].data)
        ax.set_ylim(0, max(space.data.max() for space in history) + 1)
        ax.set_xlim(0, history[0].size)
        ax.set_xlabel("Position")
        ax.set_ylabel("State")
    else:
        im = ax.imshow(history[0].data, animated=True)
        ax.set_title("CA Evolution")

    def animate(frame):
        if isinstance(history[0], Space1D):
            line.set_ydata(history[frame].data)
            ax.set_title(f"Step {frame}")
            return [line]
        else:
            im.set_array(history[frame].data)
            ax.set_title(f"Step {frame}")
            return [im]

    anim = FuncAnimation(fig, animate, frames=len(history), interval=interval, blit=True)

    if save_path:
        anim.save(save_path, writer='pillow', fps=1000//interval)

    return anim


def plot_rule_heatmap(
    rule_array: np.ndarray,
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """Visualize a CA rule as a heatmap."""
    if len(rule_array) == 64:  # EM-4/3 rule
        # Reshape to 4x4x4 for 3D neighborhood visualization
        rule_3d = rule_array.reshape(4, 4, 4)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        for left_state in range(4):
            ax = axes[left_state]
            im = ax.imshow(rule_3d[left_state], cmap='viridis')
            ax.set_title(f"Left State = {left_state}")
            ax.set_xlabel("Right State")
            ax.set_ylabel("Center State")
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
    else:
        # Generic rule visualization
        plt.figure(figsize=figsize)
        plt.plot(rule_array, 'o-')
        plt.xlabel("Rule Index")
        plt.ylabel("Output State")
        plt.title("CA Rule")
        plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
