import pytest
import numpy as np
from activation import softmax_backward, softmax_forward, relu_backward, relu_forward


def test_softmax_forward() -> None:
    activations = np.array([-1, 2, 8, 3, 0])
    expected_output = np.array(
        [0.000122227, 0.002455, 0.9904171479633, 0.00667338, 0.000332248]
    )

    difference = softmax_forward(Z=activations) - expected_output
    assert (abs(difference) < 1e-6).all()


def test_softmax_backward() -> None:
    pass


def test_relu_forward() -> None:
    activations = np.array([-1, 84, -0.1, 0, 4, 5])
    expected_output = np.array([0, 84, 0, 0, 4, 5])

    assert (relu_forward(activations) == expected_output).all()


def test_relu_backward() -> None:
    activations = np.array([-1, 84, -0.1, 0, 4, 5])
    expected_output = np.array([0, 1, 0, 0, 1, 1])

    assert (relu_backward(activations) == expected_output).all()
