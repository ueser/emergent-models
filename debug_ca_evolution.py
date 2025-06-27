"""
Jupyter Notebook Script for Debugging CA Evolution Step-by-Step

This script provides a complete setup for visualizing single genome CA evolution
in a 2D space-time diagram, perfect for debugging simulator behavior.

Usage in Jupyter:
1. Copy this entire script into a Jupyter cell
2. Run the cell to see step-by-step CA evolution
3. Modify parameters as needed for your debugging
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

# Import emergent models components
from emergent_models.core import StateModel, Tape1D
from emergent_models.encoders.em43 import Em43Encoder
from emergent_models.simulation.simulator import Simulator
from emergent_models.visualization.progression import ProgressionVisualizer
from emergent_models.visualization.tape import TapeVisualizer

# Set up matplotlib for Jupyter
%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 8)

def debug_single_genome_evolution(programme: np.ndarray, 
                                 input_val: int,
                                 rule_table: np.ndarray,
                                 max_steps: int = 20,
                                 halt_thresh: float = 0.5,
                                 window_size: int = 50,
                                 zoom_range: Optional[tuple] = None,
                                 show_step_by_step: bool = True):
    """
    Debug single genome CA evolution with step-by-step visualization.
    
    Parameters
    ----------
    programme : np.ndarray
        Programme array (e.g., [1, 2, 1])
    input_val : int
        Input value to encode
    rule_table : np.ndarray
        64-element rule lookup table
    max_steps : int, default=20
        Maximum simulation steps
    halt_thresh : float, default=0.5
        Halting threshold (fraction of blue cells)
    window_size : int, default=50
        CA tape window size
    zoom_range : tuple, optional
        (start, end) positions to focus on
    show_step_by_step : bool, default=True
        Whether to show individual step visualizations
    """
    
    print("🔬 CA Evolution Debugger")
    print("=" * 50)
    
    # Setup EM-4/3 system
    _IMMUTABLE = {
        0: 0,   # 000 -> 0
        21: 0,  # 111 -> 0  
        42: 0,  # 222 -> 0
        63: 0,  # 333 -> 0
    }
    
    state = StateModel([0,1,2,3], immutable=_IMMUTABLE)
    space = Tape1D(window_size, radius=1)
    encoder = Em43Encoder(state, space)
    simulator = Simulator(state=state, space=space, max_steps=max_steps, halt_thresh=halt_thresh)
    
    # Create visualizers
    tape_viz = TapeVisualizer(state, backend="matplotlib")
    prog_viz = ProgressionVisualizer(state, backend="matplotlib")
    
    print(f"📝 Programme: {programme}")
    print(f"📥 Input: {input_val}")
    print(f"🎯 Rule table (first 10): {rule_table[:10]}...")
    print(f"⏱️  Max steps: {max_steps}")
    print(f"🛑 Halt threshold: {halt_thresh}")
    print()
    
    # Encode initial tape
    initial_tape = encoder.encode(programme, input_val)
    print(f"🎬 Initial tape (first 20 cells): {initial_tape[:20]}")
    
    # Calculate expected positions
    prog_len = len(programme)
    separator_pos = prog_len + 2
    beacon_pos = prog_len + 2 + input_val + 1
    print(f"📍 Programme: [0:{prog_len}], Separator: [{prog_len}:{separator_pos}], Beacon: {beacon_pos}")
    print()
    
    # Run step-by-step simulation
    evolution_history = []
    current_tape = initial_tape.copy()
    evolution_history.append(current_tape.copy())
    
    halted = False
    halt_step = None
    
    print("🚀 Starting simulation...")
    for step in range(max_steps):
        # Apply one CA step
        next_tape = simulator._step_kernel(current_tape, rule_table)
        evolution_history.append(next_tape.copy())
        
        # Check halting condition
        blue_count = np.count_nonzero(next_tape == 3)
        live_count = np.count_nonzero(next_tape)
        
        if live_count > 0:
            blue_fraction = blue_count / live_count
            print(f"Step {step+1:2d}: Live={live_count:2d}, Blue={blue_count:2d}, Fraction={blue_fraction:.3f}", end="")
            
            if blue_fraction >= halt_thresh:
                print(" 🛑 HALTED!")
                halted = True
                halt_step = step + 1
                break
            else:
                print()
        else:
            print(f"Step {step+1:2d}: All cells dead ☠️")
            break
        
        current_tape = next_tape
    
    print()
    if halted:
        print(f"✅ Simulation halted at step {halt_step}")
    else:
        print(f"⏰ Simulation completed without halting ({len(evolution_history)-1} steps)")
    
    # Show individual step visualizations
    if show_step_by_step and len(evolution_history) <= 10:
        print("\n📊 Step-by-step tape states:")
        fig, axes = plt.subplots(min(len(evolution_history), 5), 1, figsize=(15, 2*min(len(evolution_history), 5)))
        if len(evolution_history) == 1:
            axes = [axes]
        
        for i, tape in enumerate(evolution_history[:5]):  # Show first 5 steps
            ax = axes[i] if i < len(axes) else None
            if ax is not None:
                # Create simple bar visualization
                colors = ['black', 'white', 'red', 'blue']
                tape_colors = [colors[state] for state in tape[:30]]  # Show first 30 cells
                
                ax.bar(range(len(tape_colors)), [1]*len(tape_colors), color=tape_colors, 
                      edgecolor='gray', linewidth=0.5)
                ax.set_title(f"Step {i}")
                ax.set_xlim(-0.5, len(tape_colors)-0.5)
                ax.set_ylim(0, 1)
                ax.set_ylabel('State')
                
                # Add annotations
                ax.axvline(prog_len-0.5, color='green', alpha=0.7, linestyle='--', label='Prog End')
                ax.axvline(separator_pos-0.5, color='orange', alpha=0.7, linestyle='--', label='Sep End')
                ax.axvline(beacon_pos, color='purple', alpha=0.7, linestyle='-', linewidth=2, label='Beacon')
                
                if i == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    # Create 2D space-time visualization
    print("\n🎨 2D Space-Time Evolution Map:")
    
    # Apply zoom if specified
    display_range = zoom_range if zoom_range else (0, min(50, len(initial_tape)))
    
    fig = prog_viz.render(
        evolution_history,
        title=f"CA Evolution: Programme={programme}, Input={input_val}",
        prog_len=prog_len,
        input_val=input_val,
        zoom_range=display_range,
        show_annotations=True,
        figsize=(15, 8)
    )
    
    plt.show()
    
    # Decode final output
    final_tape = evolution_history[-1]
    decoded_output = encoder.decode(final_tape, prog_len)
    expected_output = input_val * 2  # For doubling task
    
    print(f"\n🎯 Results:")
    print(f"   Input: {input_val}")
    print(f"   Decoded output: {decoded_output}")
    print(f"   Expected (2×input): {expected_output}")
    print(f"   Correct: {'✅' if decoded_output == expected_output else '❌'}")
    
    return evolution_history, decoded_output

# Example usage - modify these parameters for your debugging
if __name__ == "__main__":
    # Test parameters
    programme = np.array([1, 2, 1], dtype=np.uint8)
    input_val = 3
    
    # Create a simple rule table for testing
    rule_table = np.zeros(64, dtype=np.uint8)
    
    # Add some rules that preserve and transform states
    rule_table[1] = 1   # 001 -> 1 (preserve white)
    rule_table[2] = 2   # 002 -> 2 (preserve red)  
    rule_table[3] = 3   # 003 -> 3 (preserve blue)
    rule_table[6] = 2   # 012 -> 2 (white+red interaction)
    rule_table[9] = 1   # 021 -> 1 (red+white interaction)
    rule_table[18] = 2  # 102 -> 2 (spread red)
    
    # Apply immutable rules
    _IMMUTABLE = {0: 0, 21: 0, 42: 0, 63: 0}
    for k, v in _IMMUTABLE.items():
        rule_table[k] = v
    
    print("🧪 Running CA Evolution Debug Session")
    print("=" * 60)
    
    # Run the debugging session
    history, output = debug_single_genome_evolution(
        programme=programme,
        input_val=input_val,
        rule_table=rule_table,
        max_steps=15,
        halt_thresh=0.5,
        window_size=40,
        zoom_range=(0, 25),  # Focus on first 25 cells
        show_step_by_step=True
    )
    
    print(f"\n🏁 Debug session complete! Evolution had {len(history)} steps.")
