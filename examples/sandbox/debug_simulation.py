#!/usr/bin/env python3
"""
Debug CA Simulation Issues

This script investigates why the CA simulation is not performing doubling correctly.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def debug_old_simulation():
    """Debug the old implementation simulation."""
    print("🔍 Debugging OLD Implementation Simulation")
    print("=" * 50)
    
    import examples.sandbox.em43_doubling_old as old_module
    
    # Create a simple test case
    rule = np.zeros(64, dtype=np.uint8)  # Start with all-zero rule
    prog = np.array([1, 0, 2, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)  # Simple programme
    inputs = np.array([1, 2, 3], dtype=np.int64)
    
    print(f"Test rule (first 10): {rule[:10]}")
    print(f"Test programme: {prog}")
    print(f"Test inputs: {inputs}")
    
    # Apply sanitization
    rule = old_module._sanitize_rule(rule)
    prog = old_module._sanitize_programme(prog)
    
    print(f"Sanitized rule (first 10): {rule[:10]}")
    print(f"Sanitized programme: {prog}")
    
    # Test simulation step by step
    L = len(prog)
    B = len(inputs)
    N = old_module.WINDOW
    
    print(f"\nSimulation parameters:")
    print(f"Programme length (L): {L}")
    print(f"Batch size (B): {B}")
    print(f"Window size (N): {N}")
    print(f"Max steps: {old_module.MAX_STEPS}")
    print(f"Halt threshold: {old_module.HALT_THRESH}")
    
    # Manual simulation to understand what's happening
    state = np.zeros((B, N), np.uint8)
    
    # Write programme & separator
    for b in range(B):
        for j in range(L):
            state[b, j] = prog[j]
        state[b, L] = 3     # B
        state[b, L + 1] = 3     # B
    
    print(f"\nAfter writing programme and separator:")
    for b in range(B):
        print(f"Batch {b}: {state[b, :20]}...")  # Show first 20 cells
    
    # Write beacons
    for b in range(B):
        r_idx = L + 2 + inputs[b] + 1
        if r_idx < N:
            state[b, r_idx] = 2
        print(f"Batch {b}: beacon at position {r_idx}")
    
    print(f"\nAfter writing beacons:")
    for b in range(B):
        print(f"Batch {b}: {state[b, :20]}...")
    
    # Run full simulation
    outputs = old_module._simulate(rule, prog, inputs, N, old_module.MAX_STEPS, old_module.HALT_THRESH)
    print(f"\nSimulation outputs: {outputs}")
    print(f"Expected (2*inputs): {2 * inputs}")
    print(f"Errors: {np.abs(outputs - 2 * inputs)}")
    
    # Test with a random rule that might work better
    print(f"\n🎲 Testing with random rule...")
    random_rule = old_module.rng.integers(0, 4, 64, dtype=np.uint8)
    random_rule = old_module._sanitize_rule(random_rule)
    
    outputs_random = old_module._simulate(random_rule, prog, inputs, N, old_module.MAX_STEPS, old_module.HALT_THRESH)
    print(f"Random rule outputs: {outputs_random}")
    print(f"Random rule errors: {np.abs(outputs_random - 2 * inputs)}")


def debug_new_simulation():
    """Debug the new implementation simulation."""
    print("\n🔍 Debugging NEW Implementation Simulation")
    print("=" * 50)
    
    from emergent_models.core.state import StateModel
    from emergent_models.rules.sanitization import lut_idx
    from emergent_models.core.space_model import Tape1D
    from emergent_models.encoders.em43 import Em43Encoder
    from emergent_models.simulation.simulator import Simulator
    
    # Setup
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
    
    # Test encoding
    programme = np.array([1, 0, 2, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    test_input = 3
    
    print(f"Test programme: {programme}")
    print(f"Test input: {test_input}")
    
    # Test encoding
    tape = encoder.encode(programme, test_input)
    print(f"Encoded tape (first 20): {tape[:20]}")
    
    # Test simulation
    rule_table = np.zeros(64, dtype=np.uint8)
    final_tape = sim.run(tape, rule_table)
    print(f"Final tape (first 20): {final_tape[:20]}")
    
    # Test decoding
    output = encoder.decode(final_tape, len(programme), test_input)
    print(f"Decoded output: {output}")
    print(f"Expected output: {2 * test_input}")
    print(f"Error: {abs(output - 2 * test_input)}")


def analyze_encoding_scheme():
    """Analyze the EM-4/3 encoding scheme."""
    print("\n🔍 Analyzing EM-4/3 Encoding Scheme")
    print("=" * 50)
    
    # The EM-4/3 encoding should be: [programme] BB 0^(input+1) R 0...
    # For input=3: [prog] BB 0000 R 0...
    
    programme = np.array([1, 0, 2], dtype=np.uint8)
    input_val = 3
    
    print(f"Programme: {programme}")
    print(f"Input: {input_val}")
    print(f"Expected encoding: [1,0,2] [3,3] [0,0,0,0] [2] [0,0,...]")
    print(f"Positions: prog(0-2) sep(3-4) zeros(5-8) beacon(9) rest(10+)")
    
    # Manual encoding
    L = len(programme)
    tape = np.zeros(20, dtype=np.uint8)
    
    # Write programme
    tape[:L] = programme
    
    # Write separator
    tape[L:L+2] = [3, 3]
    
    # Write input+1 zeros (already zeros)
    
    # Write beacon
    beacon_pos = L + 2 + input_val + 1
    tape[beacon_pos] = 2
    
    print(f"Manual encoding: {tape}")
    print(f"Beacon position: {beacon_pos}")
    
    # For doubling, the beacon should move to position representing 2*input
    expected_output_pos = L + 2 + (2 * input_val) + 1
    print(f"Expected output beacon position: {expected_output_pos}")
    print(f"Expected output: {expected_output_pos - (L + 3)}")


def test_fitness_calculation():
    """Test fitness calculation differences."""
    print("\n🔍 Testing Fitness Calculation")
    print("=" * 50)
    
    # Test old implementation fitness
    import examples.sandbox.em43_doubling_old as old_module
    
    outputs = np.array([1, 2, 3], dtype=np.int32)
    targets = np.array([2, 4, 6], dtype=np.int32)
    programme = np.array([1, 0, 2, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    
    print(f"Test outputs: {outputs}")
    print(f"Test targets: {targets}")
    print(f"Test programme: {programme}")
    
    # Old implementation fitness calculation
    avg_err = np.abs(outputs - targets).mean()
    sparsity = np.count_nonzero(programme) / len(programme)
    old_fitness = -avg_err - old_module.LAMBDA_P * sparsity
    
    print(f"\nOld implementation:")
    print(f"Average error: {avg_err}")
    print(f"Sparsity: {sparsity}")
    print(f"Lambda_P: {old_module.LAMBDA_P}")
    print(f"Fitness: {old_fitness}")
    
    # New implementation fitness calculation
    from emergent_models.training.new_fitness import AbsoluteDifferenceFitness, ComplexityRewardFitness
    
    base_fitness = AbsoluteDifferenceFitness(continuous=True)
    fitness_fn = ComplexityRewardFitness(base_fitness, complexity_bonus=0.05)
    
    # Test new fitness
    inputs = np.array([1, 2, 3], dtype=np.int64)
    sparsities = np.array([sparsity])
    
    new_fitness = fitness_fn(outputs, inputs, sparsities=sparsities)
    
    print(f"\nNew implementation:")
    print(f"Base fitness: {base_fitness(outputs, inputs)}")
    print(f"Complexity bonus: 0.05")
    print(f"Final fitness: {new_fitness}")


if __name__ == "__main__":
    try:
        debug_old_simulation()
        debug_new_simulation()
        analyze_encoding_scheme()
        test_fitness_calculation()
        
    except Exception as e:
        print(f"\n❌ Debug failed with error: {e}")
        import traceback
        traceback.print_exc()
