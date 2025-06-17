from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt

from ..core.space import CASpace


def visualize_evolution(history: List[CASpace]) -> None:  # pragma: no cover - placeholder
    """Visualize the evolution of a cellular automaton."""
    for step, space in enumerate(history):
        plt.imshow(space.data, cmap="binary")
        plt.title(f"Step {step}")
        plt.show()
