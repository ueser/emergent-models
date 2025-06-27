#!/usr/bin/env python3
"""
Simple CA Inspector
==================

A lightweight tool to inspect individual CA genomes and understand
why they're not working.
"""

import sys
import os
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from emergent_models.core.state import StateModel
from emergent_models.core.space_model import Tape1D
from emergent_models.encoders.em43 import Em43Encoder
from emergent_models.simulation.simulator import Simulator
from emergent_models.genome import create_em43_genome


def print_tape_section(tape, start=0, end=50, title="Tape"):
    """Print a section of the tape with position markers."""
    
    section = tape[start:end]
    positions = list(range(start, min(end, len(tape))))
    
    print(f"\n{title} (positions {start}-{end-1}):")
    print("Pos: " + "".join(f"{i:2d}" for i in positions))
    print("Val: " + "".join(f"{v:2d}" for v in section))
    
    # Mark special positions
    markers = []
    for i, pos in enumerate(positions):
        if section[i] == 2:  # Red beacon
            markers.append("R ")
        elif section[i] == 3:  # Blue separator
            markers.append("B ")
        elif section[i] == 1:  # Gray
            markers.append("G ")
        else:
            markers.append("  ")
    
    print("    " + "".join(markers))


def analyze_genome(genome, input_val=3):
    """Analyze a single genome step by step."""
    
    print(f"\n🔍 ANALYZING GENOME")
    print("=" * 50)
    
    print(f"Programme: {genome.programme.code}")
    print(f"Programme length: {len(genome.programme)}")
    print(f"Programme sparsity: {genome.programme.sparsity():.2f}")
    
    # Check rule table
    rule_table = genome.rule.table
    print(f"Rule table length: {len(rule_table)}")
    print(f"Rule table sample: {rule_table[:16]}")
    
    # Check immutable constraints
    print(f"Immutable constraints: {genome.rule.state.immutable}")
    
    # Verify constraints are applied
    for lut_idx, expected_state in genome.rule.state.immutable.items():
        actual_state = rule_table[lut_idx]
        if actual_state != expected_state:
            print(f"⚠️  Constraint violation: LUT[{lut_idx}] = {actual_state}, expected {expected_state}")
        else:
            print(f"✓ Constraint OK: LUT[{lut_idx}] = {actual_state}")


def test_encoding(encoder, programme, input_val):
    """Test the encoding process."""
    
    print(f"\n🔧 TESTING ENCODING")
    print("=" * 30)
    
    tape = encoder.encode(programme, input_val)
    L = len(programme)
    
    print(f"Input: {input_val}")
    print(f"Programme length: {L}")
    
    # Expected positions
    separator_pos = [L, L+1]
    beacon_pos = L + 2 + input_val + 1
    
    print(f"Expected separator at: {separator_pos}")
    print(f"Expected beacon at: {beacon_pos}")
    
    # Check actual encoding
    print_tape_section(tape, 0, min(50, len(tape)), "Encoded tape")
    
    # Verify structure
    if L < len(tape) and tape[L] == 3 and tape[L+1] == 3:
        print("✓ Separator (BB) correctly placed")
    else:
        print(f"⚠️  Separator issue: positions {L},{L+1} = {tape[L]},{tape[L+1]}")
    
    if beacon_pos < len(tape) and tape[beacon_pos] == 2:
        print(f"✓ Beacon (R) correctly placed at position {beacon_pos}")
    else:
        print(f"⚠️  Beacon issue: position {beacon_pos} = {tape[beacon_pos] if beacon_pos < len(tape) else 'out of bounds'}")
    
    return tape


def test_simulation(simulator, tape, rule_table, max_steps=20):
    """Test the simulation process."""
    
    print(f"\n⚙️  TESTING SIMULATION")
    print("=" * 30)
    
    current_tape = tape.copy()
    
    print("Initial state:")
    print_tape_section(current_tape, 0, 50)
    
    for step in range(min(max_steps, simulator.max_steps)):
        # Apply one step
        next_tape = np.zeros_like(current_tape)
        changes = 0
        
        for x in range(1, len(current_tape) - 1):
            left = current_tape[x - 1]
            center = current_tape[x]
            right = current_tape[x + 1]
            
            idx = (left << 4) | (center << 2) | right
            if idx < len(rule_table):
                new_state = rule_table[idx]
                next_tape[x] = new_state
                if new_state != center:
                    changes += 1
        
        current_tape = next_tape
        
        print(f"\nStep {step+1}: {changes} changes")
        
        # Show state if there are changes or first few steps
        if changes > 0 or step < 3:
            print_tape_section(current_tape, 0, 50)
        
        # Check halting
        if simulator._check_halt(current_tape):
            print(f"🛑 HALTED at step {step+1}")
            break
        
        if changes == 0:
            print("🔄 No changes - static state reached")
            break
    
    return current_tape


def test_decoding(encoder, final_tape, programme_length, input_val):
    """Test the decoding process."""
    
    print(f"\n🔓 TESTING DECODING")
    print("=" * 30)
    
    print("Final tape:")
    print_tape_section(final_tape, 0, 50)
    
    # Find red beacons
    red_positions = np.where(final_tape == 2)[0]
    print(f"Red beacon positions: {red_positions}")
    
    if len(red_positions) == 0:
        print("❌ No red beacons found!")
        return -1
    
    rightmost_red = red_positions[-1]
    expected_input_beacon = programme_length + 2 + input_val + 1
    
    print(f"Rightmost red beacon: {rightmost_red}")
    print(f"Expected input beacon: {expected_input_beacon}")
    
    raw_output = rightmost_red - expected_input_beacon
    final_output = max(0, raw_output)
    
    print(f"Raw output: {rightmost_red} - {expected_input_beacon} = {raw_output}")
    print(f"Final output: {final_output}")
    
    # Use encoder's decode method
    encoder_output = encoder.decode(final_tape)
    print(f"Encoder decode: {encoder_output}")
    
    return final_output


def full_test(genome, input_val, expected_output):
    """Run a complete test of the genome."""
    
    print(f"\n🧪 FULL TEST: Input {input_val} → Expected {expected_output}")
    print("=" * 60)
    
    # Setup
    state = StateModel([0, 1, 2, 3], immutable={0: 0})
    space = Tape1D(length=200, radius=1)
    encoder = Em43Encoder(state, space)
    simulator = Simulator(state, space, max_steps=50, halt_thresh=0.5)
    
    # 1. Analyze genome
    analyze_genome(genome, input_val)
    
    # 2. Test encoding
    tape = test_encoding(encoder, genome.programme.code, input_val)
    
    # 3. Test simulation
    final_tape = test_simulation(simulator, tape, genome.rule.table)
    
    # 4. Test decoding
    output = test_decoding(encoder, final_tape, len(genome.programme), input_val)
    
    # 5. Results
    print(f"\n📊 FINAL RESULT")
    print("=" * 20)
    print(f"Input: {input_val}")
    print(f"Expected: {expected_output}")
    print(f"Actual: {output}")
    print(f"Correct: {'✅' if output == expected_output else '❌'}")
    
    return output == expected_output


def main():
    """Main inspection function."""
    
    print("🔍 Simple CA Inspector")
    print("=" * 40)
    
    # Create a test genome
    genome = create_em43_genome()
    
    # Test with a few inputs
    test_cases = [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10)]
    
    results = []
    for input_val, expected in test_cases:
        success = full_test(genome, input_val, expected)
        results.append(success)
        
        print("\n" + "="*60 + "\n")
    
    # Summary
    print(f"📈 SUMMARY")
    print("=" * 20)
    correct = sum(results)
    total = len(results)
    print(f"Correct: {correct}/{total} ({100*correct/total:.1f}%)")
    
    if correct == 0:
        print("\n💡 DEBUGGING SUGGESTIONS:")
        print("1. Check if the programme is too sparse (all zeros)")
        print("2. Verify rule table has meaningful transitions")
        print("3. Check if simulation reaches halting condition")
        print("4. Verify encoding/decoding logic")


if __name__ == "__main__":
    main()
