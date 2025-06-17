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
        inputs: List[int]
    ) -> np.ndarray:
        """
        Simulate a batch of inputs with the given genome using positional encoding.

        Parameters
        ----------
        genome : EM43Genome
            The genome containing rule and programme
        inputs : List[int]
            List of input values to simulate

        Returns
        -------
        np.ndarray
            Array of output values (-10 indicates failure)
        """
        rule_array = genome.rule.get_rule_array()
        programme = genome.programme
        inputs_array = np.asarray(inputs, dtype=np.int64)

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

    def simulate_spaces(self, genome: EM43Genome, initial_spaces: List[CASpace]) -> List[CASpace]:
        """
        Simulate a batch of initial CA spaces with the given genome.
        This is the core simulation method that is encoder-agnostic.

        Parameters
        ----------
        genome : EM43Genome
            The genome containing rule and programme
        initial_spaces : List[CASpace]
            List of initial CA spaces to simulate

        Returns
        -------
        List[CASpace]
            List of final CA spaces after simulation
        """
        rule_array = genome.rule.get_rule_array()
        final_spaces = []

        for space in initial_spaces:
            if not isinstance(space, Space1D):
                # For non-1D spaces, return unchanged for now
                final_spaces.append(space.clone())
                continue

            # Simulate this space
            current_data = space.data.copy()

            for step in range(self.max_steps):
                next_data = np.zeros_like(current_data)

                # Apply CA rules (boundary cells stay 0)
                for i in range(1, len(current_data) - 1):
                    left = current_data[i - 1]
                    center = current_data[i]
                    right = current_data[i + 1]
                    idx = (left << 4) | (center << 2) | right
                    next_data[i] = rule_array[idx]

                current_data = next_data

                # Check halting condition
                live = np.count_nonzero(current_data)
                blue = np.count_nonzero(current_data == 3)

                if live > 0 and blue / live >= self.halt_thresh:
                    break

            # Create final space
            final_space = Space1D(len(current_data), 4, space.device)
            final_space.data[:] = current_data
            final_spaces.append(final_space)

        return final_spaces


