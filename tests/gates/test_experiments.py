import pytest
import numpy as np

from pulse_opt.gates.experiments import _reshape_gate


def test_reshape_gate_trivial():
    original_gate = np.array([[1, 0], [0, 1]])
    expected_result = np.array([1, 0, 0, 1, 0, 0, 0, 0])
    result = _reshape_gate(original_gate)
    assert np.allclose(result, expected_result), \
        f"Expected {expected_result} for the input {original_gate}, but found {result}."


def test_reshape_gate_X():
    original_gate = np.array([[0, 1], [1J, 0]])
    expected_result = np.array([0, 1, 0, 0, 0, 0, 1, 0])
    result = _reshape_gate(original_gate)
    assert np.allclose(result, expected_result), \
        f"Expected {expected_result} for the input {original_gate}, but found {result}."


def test_reshape_gate_CNOT():
    original_gate = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]])
    expected_result = np.array(
        [1, 0, 0, 0] + [0, 1, 0, 0] + [0, 0, 0, 1] + [0, 0, 1, 0]
      + [0, 0, 0, 0] + [0, 0, 0, 0] + [0, 0, 0, 0] + [0, 0, 0, 0]
    )
    result = _reshape_gate(original_gate)
    assert np.allclose(result, expected_result), \
        f"Expected {expected_result} for the input {original_gate}, but found {result}."
