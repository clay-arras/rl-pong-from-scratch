import pytest
import numpy as np
from activation import softmax_backward, softmax_forward, relu_backward, relu_forward


def test_softmax_forward() -> np.ndarray[float]:
    activations = np.array([-1, 2, 8, 3, 0])
    expected_output = np.array([0.000122227, 0.002455, 0.990418, 0.00667337, 0.000332247939175495])

    difference = softmax_forward(Z=activations) - expected_output
    assert (abs(difference) < 1e-6).all()
