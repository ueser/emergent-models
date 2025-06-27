#!/usr/bin/env python3
"""
Rule Analyzer
=============

Analyze CA rules to understand what they do and suggest improvements.
"""

import sys
import os
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from emergent_models.core.state import StateModel
from emergent_models.rules.ruleset import create_em43_ruleset


def lut_idx(left, center, right):
    """Convert neighborhood to LUT index."""
    return (left << 4) | (center << 2) | right


def analyze_rule_table(rule_table):
    """Analyze what a rule table does."""
    
    print("🔍 RULE TABLE ANALYSIS")
    print("=" * 40)
    
    # Check key patterns
    patterns = {
        "Empty space": (0, 0, 0),
        "Red propagation L": (2, 0, 0),
        "Red propagation R": (0, 0, 2),
        "Red center": (0, 2, 0),
        "Red pair": (2, 2, 0),
        "Blue boundary": (0, 3, 3),
        "Blue center": (0, 3, 0),
    }
    
    print("Key patterns:")
    for name, (l, c, r) in patterns.items():
        idx = lut_idx(l, c, r)
        output = rule_table[idx]
        print(f"  {name:15s} ({l}{c}{r}) -> {output}")
    
    # Count transitions by input state
    print(f"\nTransition counts:")
    for input_state in range(4):
        count = np.sum(rule_table == input_state)
        print(f"  -> {input_state}: {count:2d} rules ({100*count/64:.1f}%)")
    
    # Check for beacon propagation
    print(f"\nBeacon propagation analysis:")
    
    # Red beacon moving right: 002 -> ?
    right_prop = rule_table[lut_idx(0, 0, 2)]
    print(f"  Red beacon right (002): -> {right_prop}")
    
    # Red beacon moving left: 200 -> ?
    left_prop = rule_table[lut_idx(2, 0, 0)]
    print(f"  Red beacon left (200):  -> {left_prop}")
    
    # Red beacon center: 020 -> ?
    center_prop = rule_table[lut_idx(0, 2, 0)]
    print(f"  Red beacon center (020): -> {center_prop}")


def suggest_doubling_rules():
    """Suggest rules that might implement doubling."""
    
    print("\n💡 DOUBLING RULE SUGGESTIONS")
    print("=" * 40)
    
    print("For doubling, you typically need:")
    print("1. Red beacons to propagate rightward")
    print("2. Mechanism to create additional beacons")
    print("3. Proper interaction with programme")
    
    print("\nSuggested key rules:")
    print("  002 -> 2  (red beacon moves right)")
    print("  020 -> 2  (red beacon stays)")
    print("  022 -> 2  (red beacons merge)")
    print("  200 -> 0  (clean up left trail)")
    print("  202 -> 2  (beacon through empty)")
    
    # Create a simple doubling rule
    rule_table = np.zeros(64, dtype=np.uint8)
    
    # Basic propagation
    rule_table[lut_idx(0, 0, 2)] = 2  # Right propagation
    rule_table[lut_idx(0, 2, 0)] = 2  # Stay in place
    rule_table[lut_idx(2, 0, 0)] = 0  # Clean left
    rule_table[lut_idx(0, 2, 2)] = 2  # Beacon interaction
    rule_table[lut_idx(2, 2, 0)] = 2  # Beacon pair
    
    # Programme interaction (example)
    rule_table[lut_idx(2, 0, 2)] = 2  # Beacon through programme
    rule_table[lut_idx(1, 2, 0)] = 2  # Programme-beacon interaction
    
    print(f"\nExample doubling rule table (first 16 entries):")
    print(f"  {rule_table[:16]}")
    
    return rule_table


def test_simple_propagation():
    """Test simple beacon propagation."""
    
    print("\n🧪 TESTING SIMPLE PROPAGATION")
    print("=" * 40)
    
    # Create simple test rule
    rule_table = np.zeros(64, dtype=np.uint8)
    rule_table[lut_idx(0, 0, 2)] = 2  # 002 -> 2 (move right)
    rule_table[lut_idx(0, 2, 0)] = 2  # 020 -> 2 (stay)
    rule_table[lut_idx(2, 0, 0)] = 0  # 200 -> 0 (clean left)
    
    # Test tape: [0, 0, 2, 0, 0, 0]
    tape = np.array([0, 0, 2, 0, 0, 0], dtype=np.uint8)
    
    print(f"Initial: {tape}")
    
    for step in range(3):
        next_tape = np.zeros_like(tape)
        
        for x in range(1, len(tape) - 1):
            left = tape[x - 1]
            center = tape[x]
            right = tape[x + 1]
            idx = lut_idx(left, center, right)
            next_tape[x] = rule_table[idx]
        
        tape = next_tape
        print(f"Step {step+1}: {tape}")


def create_working_doubling_genome():
    """Create a genome that might actually work for doubling."""
    
    print("\n🔧 CREATING WORKING DOUBLING GENOME")
    print("=" * 40)
    
    # Create a rule table designed for doubling
    rule_table = np.zeros(64, dtype=np.uint8)
    
    # Immutable constraints (required)
    immutable = {
        0: 0,   # 000 -> 0
        21: 1,  # 111 -> 1  
        42: 2,  # 222 -> 2
        63: 3,  # 333 -> 3
        60: 3,  # 330 -> 3
        3: 0,   # 003 -> 0
        48: 0,  # 300 -> 0
    }
    
    # Apply immutable constraints
    for idx, state in immutable.items():
        rule_table[idx] = state
    
    # Add doubling logic
    rule_table[lut_idx(0, 0, 2)] = 2  # Red moves right
    rule_table[lut_idx(0, 2, 0)] = 2  # Red stays
    rule_table[lut_idx(2, 0, 0)] = 0  # Clean left trail
    rule_table[lut_idx(0, 2, 2)] = 2  # Red interaction
    rule_table[lut_idx(2, 2, 0)] = 2  # Red pair
    rule_table[lut_idx(1, 0, 2)] = 2  # Programme interaction
    rule_table[lut_idx(2, 0, 1)] = 2  # Programme interaction
    
    # Create simple programme
    programme = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    
    print(f"Rule table sample: {rule_table[:16]}")
    print(f"Programme: {programme}")
    
    # Create genome
    from emergent_models.genome import Genome
    from emergent_models.rules.ruleset import RuleSet
    from emergent_models.rules.programme import Programme
    
    state = StateModel([0, 1, 2, 3], immutable=immutable)
    ruleset = RuleSet(rule_table, state)
    prog = Programme(programme, state)
    genome = Genome(ruleset, prog)
    
    return genome


def main():
    """Main analysis function."""
    
    print("🔬 Rule Analyzer")
    print("=" * 30)
    
    # Analyze a random rule
    random_genome = create_em43_ruleset()
    print("Random rule analysis:")
    analyze_rule_table(random_genome.table)
    
    # Suggest improvements
    suggest_doubling_rules()
    
    # Test simple propagation
    test_simple_propagation()
    
    # Create working genome
    working_genome = create_working_doubling_genome()
    
    print(f"\n✅ Created potentially working genome:")
    print(f"   {working_genome}")
    
    # Test it
    print(f"\n🧪 Testing working genome...")
    
    from emergent_models.core.space_model import Tape1D
    from emergent_models.encoders.em43 import Em43Encoder
    from emergent_models.simulation.simulator import Simulator
    
    space = Tape1D(length=50, radius=1)
    encoder = Em43Encoder(working_genome.rule.state, space)
    simulator = Simulator(working_genome.rule.state, space, max_steps=20, halt_thresh=0.5)
    
    # Test with input 2
    test_input = 2
    tape = encoder.encode(working_genome.programme.code, test_input)
    final_tape = simulator.run(tape, working_genome.rule.table)
    output = encoder.decode(final_tape)
    
    print(f"Input: {test_input}, Expected: {2*test_input}, Got: {output}")
    print(f"Success: {'✅' if output == 2*test_input else '❌'}")


if __name__ == "__main__":
    main()
