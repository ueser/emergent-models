#!/usr/bin/env python3
"""
Demonstration of the visualization components.

This script shows how to use the TapeVisualizer and ProgressionVisualizer
components to create beautiful CA visualizations.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from emergent_models.core.state import StateModel
from emergent_models.visualization import TapeVisualizer, ProgressionVisualizer
from emergent_models.visualization.fitness import FitnessVisualizer
from emergent_models.visualization.themes import get_scheme, list_schemes

def create_sample_data():
    """Create sample CA data for demonstration."""
    # Create a simple state model
    state_model = StateModel([0, 1, 2, 3])
    
    # Create sample tape (EM-4/3 style encoding)
    prog_len = 5
    input_val = 3
    tape_length = 30
    
    # Initial tape: [programme] [separator] [zeros] [beacon] [rest]
    tape = np.zeros(tape_length, dtype=int)
    
    # Programme
    tape[:prog_len] = [1, 0, 2, 1, 0]
    
    # Separator (BB)
    tape[prog_len:prog_len+2] = [3, 3]
    
    # Beacon at input position
    beacon_pos = prog_len + 2 + input_val + 1
    if beacon_pos < tape_length:
        tape[beacon_pos] = 2
    
    # Create evolution history (simulate some movement)
    history = [tape.copy()]
    
    # Simulate 10 steps with beacon movement
    for step in range(10):
        new_tape = tape.copy()
        
        # Move beacon slightly (simple simulation)
        current_beacon = np.where(new_tape == 2)[0]
        if len(current_beacon) > 0:
            pos = current_beacon[0]
            if pos < tape_length - 1:
                new_tape[pos] = 0
                new_tape[pos + 1] = 2
        
        # Add some random activity
        if step > 3:
            for i in range(prog_len + 2, min(tape_length, prog_len + 10)):
                if np.random.random() < 0.1:
                    new_tape[i] = np.random.choice([0, 1, 3])
        
        history.append(new_tape)
        tape = new_tape
    
    return state_model, history, prog_len, input_val


def demo_tape_visualizer():
    """Demonstrate TapeVisualizer functionality."""
    print("🎨 TapeVisualizer Demo")
    print("=" * 40)
    
    state_model, history, prog_len, input_val = create_sample_data()
    
    # Create visualizer
    tape_viz = TapeVisualizer(state_model)
    
    # Visualize initial state
    print("📊 Creating tape visualization...")
    initial_tape = history[0]
    
    try:
        fig = tape_viz.render(
            initial_tape,
            title="EM-4/3 Initial Tape State",
            prog_len=prog_len,
            input_val=input_val,
            annotate_regions=True,
            show_positions=True
        )
        
        # Save the figure
        output_dir = Path("visualization_outputs")
        output_dir.mkdir(exist_ok=True)
        
        tape_viz.save(str(output_dir / "tape_demo.png"))
        print(f"✅ Saved tape visualization to {output_dir / 'tape_demo.png'}")
        
        # Demo with zoom
        fig_zoom = tape_viz.render(
            initial_tape,
            title="EM-4/3 Tape (Zoomed)",
            prog_len=prog_len,
            input_val=input_val,
            annotate_regions=True,
            zoom_range=(0, 15)
        )
        
        tape_viz.save(str(output_dir / "tape_demo_zoom.png"))
        print(f"✅ Saved zoomed tape visualization to {output_dir / 'tape_demo_zoom.png'}")
        
    except ImportError as e:
        print(f"⚠️  Skipping matplotlib demo: {e}")
    
    # Try plotly backend if available
    try:
        tape_viz_plotly = TapeVisualizer(state_model, backend="plotly")
        fig_plotly = tape_viz_plotly.render(
            initial_tape,
            title="EM-4/3 Tape (Interactive)",
            prog_len=prog_len,
            input_val=input_val,
            annotate_regions=True
        )
        
        tape_viz_plotly.save(str(output_dir / "tape_demo_interactive.html"))
        print(f"✅ Saved interactive tape visualization to {output_dir / 'tape_demo_interactive.html'}")
        
    except ImportError as e:
        print(f"⚠️  Skipping plotly demo: {e}")


def demo_progression_visualizer():
    """Demonstrate ProgressionVisualizer functionality."""
    print("\n🎨 ProgressionVisualizer Demo")
    print("=" * 40)
    
    state_model, history, prog_len, input_val = create_sample_data()
    
    # Create visualizer
    prog_viz = ProgressionVisualizer(state_model)
    
    print("📊 Creating space-time evolution visualization...")
    
    try:
        # Static progression map
        fig = prog_viz.render(
            history,
            title="CA Space-Time Evolution",
            prog_len=prog_len,
            input_val=input_val,
            show_annotations=True
        )
        
        output_dir = Path("visualization_outputs")
        output_dir.mkdir(exist_ok=True)
        
        prog_viz.save(str(output_dir / "progression_demo.png"))
        print(f"✅ Saved progression visualization to {output_dir / 'progression_demo.png'}")
        
        # Zoomed version
        fig_zoom = prog_viz.render(
            history,
            title="CA Evolution (Zoomed)",
            prog_len=prog_len,
            input_val=input_val,
            zoom_range=(0, 20),
            show_annotations=True
        )
        
        prog_viz.save(str(output_dir / "progression_demo_zoom.png"))
        print(f"✅ Saved zoomed progression to {output_dir / 'progression_demo_zoom.png'}")
        
    except ImportError as e:
        print(f"⚠️  Skipping matplotlib demo: {e}")
    
    # Try plotly backend with animation
    try:
        prog_viz_plotly = ProgressionVisualizer(state_model, backend="plotly")
        
        # Static version
        fig_static = prog_viz_plotly.render(
            history,
            title="CA Evolution (Interactive)",
            prog_len=prog_len,
            input_val=input_val,
            show_annotations=True
        )
        
        prog_viz_plotly.save(str(output_dir / "progression_demo_interactive.html"))
        print(f"✅ Saved interactive progression to {output_dir / 'progression_demo_interactive.html'}")
        
        # Animated version
        fig_anim = prog_viz_plotly.render(
            history,
            title="CA Evolution (Animated)",
            prog_len=prog_len,
            input_val=input_val,
            animate=True,
            interval=300
        )
        
        prog_viz_plotly.save(str(output_dir / "progression_demo_animated.html"))
        print(f"✅ Saved animated progression to {output_dir / 'progression_demo_animated.html'}")
        
    except ImportError as e:
        print(f"⚠️  Skipping plotly demo: {e}")


def demo_fitness_visualizer():
    """Demonstrate FitnessVisualizer functionality."""
    print("\n🎨 FitnessVisualizer Demo")
    print("=" * 40)

    # Create sample training history data
    n_generations = 50
    generations = np.arange(n_generations)

    # Simulate realistic training progress
    np.random.seed(42)  # For reproducible results

    # Best fitness: starts low, improves with diminishing returns
    best_fitness = -20 + 15 * (1 - np.exp(-generations / 15)) + np.random.normal(0, 0.5, n_generations)
    best_fitness = np.maximum.accumulate(best_fitness)  # Ensure monotonic improvement

    # Mean fitness: follows best but with more noise and lag
    mean_fitness = best_fitness - 2 - np.random.exponential(2, n_generations)

    # Standard deviation: starts high, decreases as population converges
    std_fitness = 8 * np.exp(-generations / 20) + np.random.uniform(0.5, 2, n_generations)

    # Generation times: realistic timing with some variation
    base_time = 0.1
    gen_times = base_time + np.random.exponential(0.05, n_generations)

    training_history = {
        'best_fitness': best_fitness.tolist(),
        'mean_fitness': mean_fitness.tolist(),
        'std_fitness': std_fitness.tolist(),
        'generation_times': gen_times.tolist()
    }

    print("📊 Creating fitness evolution visualization...")

    output_dir = Path("visualization_outputs")
    output_dir.mkdir(exist_ok=True)

    try:
        # Create fitness visualizer
        fitness_viz = FitnessVisualizer()

        # Basic fitness evolution
        fig = fitness_viz.render(
            training_history,
            title="GA Training Progress",
            show_convergence=True,
            show_performance=True,
            highlight_best=True
        )

        fitness_viz.save(str(output_dir / "fitness_demo.png"))
        print(f"✅ Saved fitness visualization to {output_dir / 'fitness_demo.png'}")

        # Convergence analysis
        convergence_analysis = fitness_viz.analyze_convergence(training_history)
        print(f"📈 Convergence Analysis:")
        print(f"   Total improvement: {convergence_analysis['total_improvement']:.4f}")
        print(f"   Final fitness: {convergence_analysis['final_fitness']:.4f}")
        print(f"   Converged: {convergence_analysis['converged']}")
        if convergence_analysis['convergence_generation']:
            print(f"   Convergence generation: {convergence_analysis['convergence_generation']}")

    except ImportError as e:
        print(f"⚠️  Skipping matplotlib demo: {e}")

    # Try plotly backend
    try:
        fitness_viz_plotly = FitnessVisualizer(backend="plotly")

        fig_plotly = fitness_viz_plotly.render(
            training_history,
            title="GA Training Progress (Interactive)",
            show_convergence=True,
            show_performance=True
        )

        fitness_viz_plotly.save(str(output_dir / "fitness_demo_interactive.html"))
        print(f"✅ Saved interactive fitness visualization to {output_dir / 'fitness_demo_interactive.html'}")

    except ImportError as e:
        print(f"⚠️  Skipping plotly demo: {e}")


def demo_color_schemes():
    """Demonstrate different color schemes."""
    print("\n🎨 Color Schemes Demo")
    print("=" * 40)
    
    state_model, history, prog_len, input_val = create_sample_data()
    initial_tape = history[0]
    
    print("Available color schemes:", list_schemes())
    
    output_dir = Path("visualization_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Try different color schemes
    for scheme_name in ['em43', 'scientific', 'high_contrast']:
        try:
            scheme = get_scheme(scheme_name)
            tape_viz = TapeVisualizer(state_model, color_scheme=scheme.mapping)
            
            fig = tape_viz.render(
                initial_tape,
                title=f"Tape with {scheme.name} Colors",
                prog_len=prog_len,
                input_val=input_val,
                annotate_regions=True
            )
            
            filename = f"tape_demo_{scheme_name}.png"
            tape_viz.save(str(output_dir / filename))
            print(f"✅ Saved {scheme.name} scheme to {output_dir / filename}")
            
        except ImportError:
            print(f"⚠️  Skipping {scheme_name} scheme demo (matplotlib not available)")
        except Exception as e:
            print(f"❌ Error with {scheme_name} scheme: {e}")


def main():
    """Run all visualization demos."""
    print("🚀 Emergent Models Visualization Demo")
    print("=" * 50)
    
    try:
        demo_tape_visualizer()
        demo_progression_visualizer()
        demo_fitness_visualizer()
        demo_color_schemes()
        
        print("\n🎉 Demo completed successfully!")
        print("📁 Check the 'visualization_outputs' directory for generated files.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
