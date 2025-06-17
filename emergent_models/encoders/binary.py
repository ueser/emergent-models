from __future__ import annotations

from typing import List, Union

import numpy as np

from .base import CATransform
from ..core.space import CASpace, Space1D


class BinaryEncoder(CATransform):
    """Encode data to binary states."""

    def __call__(self, data: Union[str, List[int]]) -> CASpace:
        if isinstance(data, str):
            bits = [int(b) for b in data]
        else:
            bits = [int(x) for x in data]
        space = Space1D(size=len(bits), n_states=2)
        space.data[:] = np.array(bits, dtype=np.int32)
        return space


class EM43BinaryEncoder(CATransform):
    """
    Binary encoder for EM-4/3 CA with state mapping.

    Converts integers to binary representation, then maps:
    - 0 bits -> empty state (0)
    - 1 bits -> input state (configurable, default 2 for red)

    Example: 9 -> binary 1001 -> states [2, 0, 0, 2] (with input_state=2)
    """

    def __init__(self, bit_width: int = 8, input_state: int = 2, pad_left: bool = True):
        """
        Parameters
        ----------
        bit_width : int
            Number of bits to use for binary representation
        input_state : int
            CA state to use for '1' bits (default 2 = red beacon)
        pad_left : bool
            Whether to pad with zeros on the left (MSB first) or right
        """
        self.bit_width = bit_width
        self.input_state = input_state
        self.pad_left = pad_left

    def encode_number(self, number: int) -> np.ndarray:
        """Convert a single number to binary CA states"""
        if number < 0:
            raise ValueError(f"Cannot encode negative number: {number}")

        if number >= 2**self.bit_width:
            raise ValueError(f"Number {number} too large for {self.bit_width} bits")

        # Convert to binary
        binary_str = format(number, f'0{self.bit_width}b')

        # Convert to CA states
        states = np.zeros(self.bit_width, dtype=np.uint8)
        for i, bit in enumerate(binary_str):
            if bit == '1':
                states[i] = self.input_state
            # '0' bits remain as state 0 (empty)

        return states

    def decode_states(self, states: np.ndarray) -> int:
        """Convert CA states back to integer"""
        binary_str = ""
        for state in states:
            if state == self.input_state:
                binary_str += "1"
            else:
                binary_str += "0"

        return int(binary_str, 2) if binary_str else 0

    def __call__(self, data: Union[int, List[int]]) -> CASpace:
        """Encode integer(s) to binary CA space"""
        if isinstance(data, int):
            encoded = self.encode_number(data)
            space = Space1D(size=len(encoded), n_states=4)
            space.data[:] = encoded
            return space
        else:
            # Multiple numbers - concatenate their encodings
            all_encoded = []
            for num in data:
                encoded = self.encode_number(num)
                all_encoded.extend(encoded)

            space = Space1D(size=len(all_encoded), n_states=4)
            space.data[:] = np.array(all_encoded, dtype=np.uint8)
            return space


def int_to_binary_states(number: int, bit_width: int = 8, input_state: int = 2) -> np.ndarray:
    """
    Utility function to convert integer to binary CA states.

    Parameters
    ----------
    number : int
        Integer to convert
    bit_width : int
        Number of bits for binary representation
    input_state : int
        CA state for '1' bits (default 2 = red beacon)

    Returns
    -------
    np.ndarray
        Array of CA states representing the binary number

    Example
    -------
    >>> int_to_binary_states(9, 8, 2)  # 9 = 00001001 in binary
    array([0, 0, 0, 0, 2, 0, 0, 2], dtype=uint8)
    """
    encoder = EM43BinaryEncoder(bit_width, input_state)
    return encoder.encode_number(number)


def binary_states_to_int(states: np.ndarray, input_state: int = 2) -> int:
    """
    Utility function to convert binary CA states back to integer.

    Parameters
    ----------
    states : np.ndarray
        Array of CA states
    input_state : int
        CA state that represents '1' bits

    Returns
    -------
    int
        Decoded integer value
    """
    encoder = EM43BinaryEncoder(len(states), input_state)
    return encoder.decode_states(states)
