"""
EM-4/3 encoder for cellular automata.

This module provides the Em43Encoder that implements the standard
EM-4/3 positional encoding scheme.
"""

import numpy as np

import numba as nb

from ..core.base import Encoder
from ..core.state import StateModel
from ..core.space_model import SpaceModel


class Em43Encoder(Encoder):
    """
    EM-4/3 positional encoder.
    
    Implements the standard EM-4/3 encoding scheme:
    [programme] BB 0^(input+1) R 0...
    
    Where:
    - programme: The CA programme
    - BB: Blue separators (state 3)
    - 0^(input+1): input+1 zeros
    - R: Red beacon (state 2)
    
    Examples
    --------
    >>> from emergent_models.core import StateModel, Tape1D
    >>> state = StateModel([0,1,2,3])
    >>> space = Tape1D(200, radius=1)
    >>> encoder = Em43Encoder(state, space)
    >>> 
    >>> programme = np.array([1, 0, 2], dtype=np.uint8)
    >>> tape = encoder.encode(programme, input_val=5)
    >>> # tape: [1, 0, 2, 3, 3, 0, 0, 0, 0, 0, 2, 0, ...]
    """
    
    def __init__(self, state: StateModel, space: SpaceModel):
        """
        Initialize EM-4/3 encoder.

        Parameters
        ----------
        state : StateModel
            Must have states [0,1,2,3] for EM-4/3
        space : SpaceModel
            Space model (typically Tape1D)
        """
        super().__init__(state, space)
        
        # Validate state model for EM-4/3
        if state.n_states != 4 or state.symbols != [0,1,2,3]:
            raise ValueError("EM-4/3 encoder requires StateModel([0,1,2,3])")
    
    def encode(self, programme: np.ndarray, inp: int) -> np.ndarray:
        """
        Encode single programme and single input into initial CA tape.

        This method follows the training paradigm where:
        - Input data: (B, D1) - batch of inputs
        - Programs: (P, L) - population of programs
        - Encoded tapes: (P, T) - population of encoded tapes

        For training, this is called P times (once per program) to create
        the full population of encoded tapes (P, T).

        Parameters
        ----------
        programme : np.ndarray, shape (L,)
            Single programme array (should not contain blue cells)
        inp : int
            Single input value to encode positionally

        Returns
        -------
        np.ndarray, shape (T,)
            Single initial CA tape: [programme] BB 0^(inp+1) R 0...
        """
        if inp < 0:
            raise ValueError("Input must be non-negative")

        window = self.space.size_hint()
        L = len(programme)

        # Check if encoding fits in window
        required_size = L + 2 + inp + 1 + 1  # prog + BB + zeros + R
        if required_size > window:
            raise ValueError(f"Window size {window} too small for programme length {L} and input {inp}")

        # Use Numba-compiled single encoding
        return _encode_em43_numba(programme, inp, window)

    def encode_population(self, programmes: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Encode population of programmes with population of inputs.

        This follows the P = B paradigm where population size equals batch size.
        Each program is paired with one input.

        Parameters
        ----------
        programmes : np.ndarray, shape (P, L)
            Population of programmes
        inputs : np.ndarray, shape (P,) or (P, D1)
            Population of inputs (same size as programmes)

        Returns
        -------
        np.ndarray, shape (P, T)
            Encoded tapes for each program-input pair
        """
        # Flatten inputs if needed
        if inputs.ndim > 1:
            inputs_flat = inputs.flatten()
        else:
            inputs_flat = inputs

        P, L = programmes.shape
        assert len(inputs_flat) == P, f"Population size {P} must equal input batch size {len(inputs_flat)}"

        window = self.space.size_hint()

        # Create output tensor (P, T)
        encoded_tapes = np.zeros((P, window), dtype=np.uint8)

        # Encode each program with its corresponding input
        for p in range(P):
            encoded_tapes[p] = self.encode(programmes[p], inputs_flat[p])

        return encoded_tapes
    
    def decode(self, tape: np.ndarray, programme_length: int) -> int:
        """
        Decode single final CA tape to output value using EXACT EM-4/3 decoding.

        This method follows the training paradigm where decoding is done
        on individual tapes from the population.

        The EM-4/3 decoding formula is:
        output = rightmost_red_position - (programme_length + 3)

        Where +3 accounts for: separator (2 cells) + 1 zero before input beacon

        Parameters
        ----------
        tape : np.ndarray, shape (T,)
            Single final CA tape after simulation
        programme_length : int
            Length of the programme (needed for exact decoding)

        Returns
        -------
        int
            Decoded output value, or -1 if no valid beacon found
        """
        if tape.ndim != 1:
            raise ValueError(f"Expected 1D tape array, got {tape.ndim}D. Use decode_population() for batch decoding.")

        return _decode_em43_numba(tape, programme_length)

    def decode_population(self, tapes: np.ndarray, programme_length: int) -> np.ndarray:
        """
        Decode population of final CA tapes to output values.

        This follows the P = B paradigm where we decode P tapes to P outputs.

        Parameters
        ----------
        tapes : np.ndarray, shape (P, T)
            Population of final CA tapes after simulation
        programme_length : int
            Length of the programme (needed for exact decoding)

        Returns
        -------
        np.ndarray, shape (P,1)
            Decoded output values for each tape
        """
        if tapes.ndim != 2:
            raise ValueError(f"Expected 2D tape array (P, T), got {tapes.ndim}D")

        P = tapes.shape[0]
        outputs = np.zeros((P,1), dtype=np.int32)
        for p in range(P):
            outputs[p,0] = self.decode(tapes[p], programme_length)
        return outputs



@nb.njit(inline='always', cache=True)
def _encode_em43_numba(programme: np.ndarray, inp: int, window: int) -> np.ndarray:
    """Numba-optimized EM-4/3 encoding for single input."""
    L = len(programme)
    tape = np.zeros(window, dtype=np.uint8)

    # Write programme
    for i in range(L):
        tape[i] = programme[i]

    # Write separator (BB)
    if L < window:
        tape[L] = 3
    if L + 1 < window:
        tape[L + 1] = 3

    # Write input beacon
    beacon_pos = L + 2 + inp + 1
    if beacon_pos < window:
        tape[beacon_pos] = 2

    return tape

@nb.njit(inline='always', cache=True)
def _decode_em43_numba(tape: np.ndarray, programme_length: int) -> int:
    """
    Numba-optimized EXACT EM-4/3 decoding for single tape.

    Formula: output = rightmost_red_position - (programme_length + 3)
    Where +3 = separator (2 cells) + 1 zero before input beacon
    """
    # Find rightmost red beacon
    rightmost_red = -1
    for i in range(len(tape) - 1, -1, -1):
        if tape[i] == 2:
            rightmost_red = i
            break

    if rightmost_red == -1:
        return -1

    # EXACT EM-4/3 decoding formula
    output = rightmost_red - (programme_length + 3)
    return max(0, output)

@nb.njit(inline='always', cache=True)
def _decode_em43_batch_numba(tapes: np.ndarray, programme_length: int) -> np.ndarray:
    """
    Numba-optimized EXACT EM-4/3 batch decoding for multiple tapes.

    Formula: output = rightmost_red_position - (programme_length + 3)
    Where +3 = separator (2 cells) + 1 zero before input beacon
    """
    n_tapes = tapes.shape[0]
    outputs = np.full(n_tapes, -1, dtype=np.int32)

    for batch_idx in range(n_tapes):
        tape = tapes[batch_idx]

        # Find rightmost red beacon
        rightmost_red = -1
        for i in range(len(tape) - 1, -1, -1):
            if tape[i] == 2:
                rightmost_red = i
                break

        if rightmost_red != -1:
            # EXACT EM-4/3 decoding formula
            output = rightmost_red - (programme_length + 3)
            outputs[batch_idx] = max(0, output)

    return outputs


