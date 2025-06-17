"""
Numba-accelerated EM-4/3 Simulator
==================================
High-performance batch simulator using Numba JIT compilation.
Based on the EM-4/3 simulation patterns.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

try:
    import numba as nb
    NUMBA_AVAILABLE = True
    
    # Configure Numba to use all available threads
    nb.set_num_threads(nb.config.NUMBA_NUM_THREADS)
    print(f"Numba using {nb.get_num_threads()} threads")
    
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available, falling back to pure NumPy implementation")

from ..core.space import CASpace, Space1D
from ..rules.em43 import EM43Genome, lut_idx
from ..encoders.binary import int_to_binary_states, binary_states_to_int
from .base import Simulator


if NUMBA_AVAILABLE:
    @nb.njit(cache=True)
    def _simulate_batch_numba(
        rule_array: np.ndarray,
        programme: np.ndarray,
        inputs: np.ndarray,
        window: int,
        max_steps: int,
        halt_thresh: float
    ) -> np.ndarray:
        """
        Numba-accelerated batch simulation for EM-4/3.
        
        Parameters
        ----------
        rule_array : (64,) uint8 - The CA rule lookup table
        programme : (L,) uint8 - The programme sequence
        inputs : (B,) int64 - Input values for each batch item
        window : int - Tape length
        max_steps : int - Maximum simulation steps
        halt_thresh : float - Halting threshold (fraction of blue cells)
        
        Returns
        -------
        outputs : (B,) int32 - Output values (-10 on failure)
        """
        L = programme.shape[0]
        B = inputs.shape[0]
        N = window
        
        # Initialize state arrays
        state = np.zeros((B, N), np.uint8)
        halted = np.zeros(B, np.bool_)
        frozen = np.zeros_like(state)
        output = np.full(B, -10, np.int32)
        
        # Setup initial tape for each batch item
        for b in range(B):
            # Write programme
            for j in range(L):
                state[b, j] = programme[j]
            
            # Write separator (BB)
            state[b, L] = 3      # Blue
            state[b, L + 1] = 3  # Blue
            
            # Write beacon pattern: 0^(input+1) R 0
            beacon_pos = L + 2 + inputs[b] + 1
            if beacon_pos < N:
                state[b, beacon_pos] = 2  # Red beacon
        
        # Main simulation loop
        for step in range(max_steps):
            active_any = False
            
            for b in range(B):
                if halted[b]:
                    continue
                    
                active_any = True
                
                # Apply CA rules
                next_state = np.zeros(N, np.uint8)
                
                # Boundary cells stay 0
                next_state[0] = 0
                next_state[N-1] = 0
                
                # Apply rules to interior cells
                for x in range(1, N - 1):
                    left = state[b, x - 1]
                    center = state[b, x]
                    right = state[b, x + 1]
                    
                    # Lookup table index
                    idx = (left << 4) | (center << 2) | right
                    next_state[x] = rule_array[idx]
                
                state[b] = next_state
                
                # Check halting condition
                live_count = 0
                blue_count = 0
                
                for x in range(N):
                    cell_val = next_state[x]
                    if cell_val != 0:
                        live_count += 1
                        if cell_val == 3:  # Blue
                            blue_count += 1
                
                # Halt if enough blue cells
                if live_count > 0 and blue_count / live_count >= halt_thresh:
                    halted[b] = True
                    frozen[b] = next_state
            
            if not active_any:
                break
        
        # Decode outputs
        for b in range(B):
            if not halted[b]:
                continue
                
            # Find rightmost red beacon
            rightmost_red = -1
            for x in range(N - 1, -1, -1):
                if frozen[b, x] == 2:  # Red
                    rightmost_red = x
                    break
            
            if rightmost_red != -1:
                # Calculate output relative to expected beacon position
                expected_beacon = L + 2 + inputs[b] + 1
                output[b] = rightmost_red - expected_beacon
        
        return output

    @nb.njit(cache=True)
    def _simulate_batch_binary_numba(
        rule_array: np.ndarray,
        programme: np.ndarray,
        inputs: np.ndarray,
        window: int,
        max_steps: int,
        halt_thresh: float,
        bit_width: int,
        input_state: int
    ) -> np.ndarray:
        """
        Numba-accelerated binary simulation for EM-4/3.

        Tape structure: [programme] BB [binary_input] 0...
        """
        L = programme.shape[0]
        B = inputs.shape[0]
        N = window

        # Initialize state arrays
        state = np.zeros((B, N), np.uint8)
        halted = np.zeros(B, np.bool_)
        frozen = np.zeros_like(state)
        output = np.full(B, -10, np.int32)

        # Setup initial tape for each batch item
        for b in range(B):
            # Write programme
            for j in range(L):
                state[b, j] = programme[j]

            # Write separator (BB)
            state[b, L] = 3      # Blue
            state[b, L + 1] = 3  # Blue

            # Write binary input: convert number to binary states
            input_num = inputs[b]
            for bit_pos in range(bit_width):
                bit_val = (input_num >> (bit_width - 1 - bit_pos)) & 1
                if bit_val == 1:
                    state[b, L + 2 + bit_pos] = input_state
                # 0 bits remain as state 0

        # Main simulation loop
        for step in range(max_steps):
            active_any = False

            for b in range(B):
                if halted[b]:
                    continue

                active_any = True

                # Apply CA rules
                next_state = np.zeros(N, np.uint8)

                # Boundary cells stay 0
                next_state[0] = 0
                next_state[N-1] = 0

                # Apply rules to interior cells
                for x in range(1, N - 1):
                    left = state[b, x - 1]
                    center = state[b, x]
                    right = state[b, x + 1]

                    # Lookup table index
                    idx = (left << 4) | (center << 2) | right
                    next_state[x] = rule_array[idx]

                state[b] = next_state

                # Check halting condition
                live_count = 0
                blue_count = 0

                for x in range(N):
                    cell_val = next_state[x]
                    if cell_val != 0:
                        live_count += 1
                        if cell_val == 3:  # Blue
                            blue_count += 1

                # Halt if enough blue cells
                if live_count > 0 and blue_count / live_count >= halt_thresh:
                    halted[b] = True
                    frozen[b] = next_state

            if not active_any:
                break

        # Decode outputs - look for binary patterns
        for b in range(B):
            if not halted[b]:
                continue

            # Find the rightmost cluster of input_state cells
            # that could represent a binary number
            best_result = -1

            # Scan from right to left looking for bit patterns
            for start_pos in range(N - bit_width, -1, -1):
                # Check if we have a valid bit pattern here
                binary_val = 0
                valid_pattern = True

                for bit_pos in range(bit_width):
                    cell_state = frozen[b, start_pos + bit_pos]
                    if cell_state == input_state:
                        binary_val |= (1 << (bit_width - 1 - bit_pos))
                    elif cell_state != 0:
                        # Non-zero, non-input_state cell breaks the pattern
                        valid_pattern = False
                        break

                if valid_pattern and binary_val > 0:
                    best_result = binary_val
                    break

            output[b] = best_result

        return output


def _simulate_batch_numpy(
    rule_array: np.ndarray,
    programme: np.ndarray,
    inputs: np.ndarray,
    window: int,
    max_steps: int,
    halt_thresh: float
) -> np.ndarray:
    """Pure NumPy fallback implementation"""
    # This is a simplified version for when Numba is not available
    B = len(inputs)
    outputs = np.full(B, -10, dtype=np.int32)
    
    for b in range(B):
        # Create individual simulation (simplified)
        space = Space1D(window, 4)
        
        # Set up programme and input
        L = len(programme)
        space.data[:L] = programme
        space.data[L:L+2] = 3  # Separator
        
        beacon_pos = L + 2 + inputs[b] + 1
        if beacon_pos < window:
            space.data[beacon_pos] = 2
        
        # Simple simulation loop
        for step in range(max_steps):
            new_data = np.zeros_like(space.data)
            
            for i in range(1, window - 1):
                left = space.data[i - 1]
                center = space.data[i]
                right = space.data[i + 1]
                idx = (left << 4) | (center << 2) | right
                new_data[i] = rule_array[idx]
            
            space.data = new_data
            
            # Simple halting check
            live = np.count_nonzero(space.data)
            blue = np.count_nonzero(space.data == 3)
            
            if live > 0 and blue / live >= halt_thresh:
                # Find rightmost red
                red_positions = np.where(space.data == 2)[0]
                if len(red_positions) > 0:
                    rightmost_red = red_positions[-1]
                    expected = L + 2 + inputs[b] + 1
                    outputs[b] = rightmost_red - expected
                break
    
    return outputs


class NumbaSimulator(Simulator):
    """High-performance Numba-accelerated simulator for EM-4/3"""
    
    def __init__(self, window: int = 500, max_steps: int = 256, halt_thresh: float = 0.50):
        super().__init__(max_steps)
        self.window = window
        self.halt_thresh = halt_thresh
        self.use_numba = NUMBA_AVAILABLE
    
    def simulate_batch(
        self,
        genome: EM43Genome,
        inputs: List[int],
        use_binary_encoding: bool = False,
        bit_width: int = 8,
        input_state: int = 2
    ) -> np.ndarray:
        """
        Simulate a batch of inputs with the given genome.

        Parameters
        ----------
        genome : EM43Genome
            The genome containing rule and programme
        inputs : List[int]
            List of input values to simulate
        use_binary_encoding : bool
            If True, encode inputs as binary patterns instead of positional
        bit_width : int
            Number of bits for binary encoding (only used if use_binary_encoding=True)
        input_state : int
            CA state to use for '1' bits in binary encoding (default 2 = red)

        Returns
        -------
        np.ndarray
            Array of output values (-10 indicates failure)
        """
        rule_array = genome.rule.get_rule_array()
        programme = genome.programme
        inputs_array = np.asarray(inputs, dtype=np.int64)

        if use_binary_encoding:
            return self._simulate_batch_binary(
                rule_array, programme, inputs_array,
                bit_width, input_state
            )
        else:
            if self.use_numba:
                return _simulate_batch_numba(
                    rule_array, programme, inputs_array,
                    self.window, self.max_steps, self.halt_thresh
                )
            else:
                return _simulate_batch_numpy(
                    rule_array, programme, inputs_array,
                    self.window, self.max_steps, self.halt_thresh
                )
    
    def forward(self, genome, initial_state, halting_condition=None):
        """Single simulation (for compatibility with base class)"""
        if not isinstance(genome, EM43Genome):
            return super().forward(genome, initial_state, halting_condition)
        
        # Extract input from initial state (simplified)
        if isinstance(initial_state, Space1D) and len(initial_state.data) > 0:
            input_val = int(initial_state.data[0])
            outputs = self.simulate_batch(genome, [input_val])
            
            # Create output space
            result_space = Space1D(1, 4)
            result_space.data[0] = max(0, outputs[0]) if outputs[0] >= 0 else 0
            return result_space
        
        return initial_state.clone()

    def _simulate_batch_binary(
        self,
        rule_array: np.ndarray,
        programme: np.ndarray,
        inputs: np.ndarray,
        bit_width: int,
        input_state: int
    ) -> np.ndarray:
        """
        Simulate batch with binary encoding of inputs.

        Instead of positional encoding (0^n R), uses binary encoding where
        each input number is converted to binary and 1-bits become input_state.

        Tape structure: [programme] BB [binary_input] 0...
        """
        if self.use_numba:
            return _simulate_batch_binary_numba(
                rule_array, programme, inputs,
                self.window, self.max_steps, self.halt_thresh,
                bit_width, input_state
            )
        else:
            # Fallback NumPy implementation
            B = len(inputs)
            outputs = np.full(B, -10, dtype=np.int32)

            for b in range(B):
                # Encode input as binary states
                input_binary = int_to_binary_states(inputs[b], bit_width, input_state)

                # Create tape: programme + separator + binary input
                L = len(programme)
                tape_length = L + 2 + bit_width + 50  # Extra space for computation

                if tape_length > self.window:
                    continue  # Skip if too large

                # Initialize tape
                tape = np.zeros(tape_length, dtype=np.uint8)

                # Set programme
                tape[:L] = programme

                # Set separator (BB)
                tape[L:L+2] = 3

                # Set binary input
                tape[L+2:L+2+bit_width] = input_binary

                # Simple simulation for binary case
                current_tape = tape.copy()

                for step in range(self.max_steps):
                    next_tape = np.zeros_like(current_tape)

                    # Apply CA rules
                    for i in range(1, len(current_tape) - 1):
                        left = current_tape[i - 1]
                        center = current_tape[i]
                        right = current_tape[i + 1]
                        idx = (left << 4) | (center << 2) | right
                        next_tape[i] = rule_array[idx]

                    current_tape = next_tape

                    # Check halting condition
                    live = np.count_nonzero(current_tape)
                    blue = np.count_nonzero(current_tape == 3)

                    if live > 0 and blue / live >= self.halt_thresh:
                        # Try to decode output from final tape
                        # Look for rightmost cluster of input_state
                        rightmost_cluster = []
                        for i in range(len(current_tape) - 1, -1, -1):
                            if current_tape[i] == input_state:
                                rightmost_cluster.insert(0, i)
                            elif len(rightmost_cluster) > 0:
                                break

                        if len(rightmost_cluster) >= bit_width:
                            # Extract bit pattern
                            result_states = current_tape[rightmost_cluster[:bit_width]]
                            try:
                                outputs[b] = binary_states_to_int(result_states, input_state)
                            except:
                                outputs[b] = -1
                        break

            return outputs

    def _simulate_batch_binary(
        self,
        rule_array: np.ndarray,
        programme: np.ndarray,
        inputs: np.ndarray,
        bit_width: int,
        input_state: int
    ) -> np.ndarray:
        """
        Simulate batch with binary encoding of inputs.

        Instead of positional encoding (0^n R), uses binary encoding where
        each input number is converted to binary and 1-bits become input_state.

        Tape structure: [programme] BB [binary_input] 0...
        """
        B = len(inputs)
        outputs = np.full(B, -10, dtype=np.int32)

        for b in range(B):
            # Encode input as binary states
            input_binary = int_to_binary_states(inputs[b], bit_width, input_state)

            # Create tape: programme + separator + binary input
            L = len(programme)
            tape_length = L + 2 + bit_width + 50  # Extra space for computation

            if tape_length > self.window:
                continue  # Skip if too large

            # Initialize tape
            tape = np.zeros(tape_length, dtype=np.uint8)

            # Set programme
            tape[:L] = programme

            # Set separator (BB)
            tape[L:L+2] = 3

            # Set binary input
            tape[L+2:L+2+bit_width] = input_binary

            # Simple simulation for binary case
            current_tape = tape.copy()

            for step in range(self.max_steps):
                next_tape = np.zeros_like(current_tape)

                # Apply CA rules
                for i in range(1, len(current_tape) - 1):
                    left = current_tape[i - 1]
                    center = current_tape[i]
                    right = current_tape[i + 1]
                    idx = (left << 4) | (center << 2) | right
                    next_tape[i] = rule_array[idx]

                current_tape = next_tape

                # Check halting condition
                live = np.count_nonzero(current_tape)
                blue = np.count_nonzero(current_tape == 3)

                if live > 0 and blue / live >= self.halt_thresh:
                    # Try to decode output from final tape
                    # Look for patterns that might represent the result

                    # Simple heuristic: find rightmost cluster of input_state
                    rightmost_cluster = []
                    for i in range(len(current_tape) - 1, -1, -1):
                        if current_tape[i] == input_state:
                            rightmost_cluster.insert(0, i)
                        elif len(rightmost_cluster) > 0:
                            break

                    if len(rightmost_cluster) >= bit_width:
                        # Extract bit pattern
                        result_states = current_tape[rightmost_cluster[:bit_width]]
                        try:
                            outputs[b] = binary_states_to_int(result_states, input_state)
                        except:
                            outputs[b] = -1
                    break

        return outputs
