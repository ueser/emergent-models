from __future__ import annotations

from typing import List

import numpy as np

from .base import CATransform
from ..core.space import CASpace, Space1D


class PositionEncoder(CATransform):
    """Encode data using position indexing."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __call__(self, data: List[int]) -> CASpace:
        space = Space1D(size=len(data), n_states=self.vocab_size)
        space.data[:] = np.array(data, dtype=np.int32)
        return space


class EM43PositionalEncoder(CATransform):
    """
    Positional encoder for EM-4/3 CA (original method).

    Uses positional encoding: input N becomes N zeros followed by a red beacon.
    Example: 5 -> [0,0,0,0,0,2] (5 zeros + red beacon)

    This encoder handles the complete workflow:
    1. encode_input() - creates initial CA space with programme + encoded input
    2. decode_output() - extracts output from final CA space
    """

    def __init__(self, beacon_state: int = 2, separator_state: int = 3):
        """
        Parameters
        ----------
        beacon_state : int
            CA state to use for beacons (default 2 = red)
        separator_state : int
            CA state to use for separators (default 3 = blue)
        """
        self.beacon_state = beacon_state
        self.separator_state = separator_state

    def encode_input(self, programme: np.ndarray, input_value: int, window_size: int = 200) -> CASpace:
        """
        Create initial CA space with programme and positionally encoded input.

        Tape structure: [programme] BB 0^(input+1) R 0...

        Parameters
        ----------
        programme : np.ndarray
            The CA programme
        input_value : int
            Input number to encode
        window_size : int
            Total size of the CA space

        Returns
        -------
        CASpace
            Initial CA space ready for simulation
        """
        # Calculate required space
        L = len(programme)
        required_size = L + 2 + input_value + 1 + 1  # programme + separator + zeros + beacon

        if required_size > window_size:
            raise ValueError(f"Window size {window_size} too small for programme length {L} and input {input_value}")

        # Create space
        space = Space1D(window_size, n_states=4)

        # Set programme
        space.data[:L] = programme

        # Set separator (BB)
        space.data[L:L+2] = self.separator_state

        # Set beacon at position L + 2 + input_value + 1
        beacon_pos = L + 2 + input_value + 1
        space.data[beacon_pos] = self.beacon_state

        # Zeros are already set by default
        return space

    def decode_output(self, final_space: CASpace, programme_length: int, input_value: int) -> int:
        """
        Decode output from final CA space.

        Looks for the rightmost red beacon and calculates its position relative to expected input beacon.

        Parameters
        ----------
        final_space : CASpace
            Final CA space after simulation
        programme_length : int
            Length of the programme
        input_value : int
            Original input value (to calculate expected beacon position)

        Returns
        -------
        int
            Decoded output value, or -1 if no valid beacon found
        """
        if not isinstance(final_space, Space1D):
            return -1

        data = final_space.data
        L = programme_length

        # Find rightmost red beacon
        rightmost_beacon = -1
        for i in range(len(data) - 1, -1, -1):
            if data[i] == self.beacon_state:
                rightmost_beacon = i
                break

        if rightmost_beacon == -1:
            return -1

        # Calculate output relative to expected input beacon position
        expected_input_beacon = L + 2 + input_value + 1
        output_value = rightmost_beacon - expected_input_beacon

        return max(0, output_value)  # Ensure non-negative

    def __call__(self, data: List[int]) -> CASpace:
        """Legacy method for compatibility"""
        if len(data) == 1:
            # Single input - create a simple positional space
            input_val = data[0]
            space = Space1D(input_val + 2, n_states=4)
            space.data[input_val + 1] = self.beacon_state  # Place beacon after zeros
            return space
        else:
            # Multiple inputs - not well-defined for positional encoding
            raise NotImplementedError("Positional encoding with multiple inputs not supported")
