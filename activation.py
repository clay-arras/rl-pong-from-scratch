import numpy as np
from typeguard import typechecked


@typechecked
def softmax_forward(Z: np.ndarray) -> np.ndarray:
    """
    Gives an array of softmax probabilities between 0 and 1.

    Computes with numerical stability by subtracting the max of Z.

    Parameters:
    Z (np.ndarray): The activations, shape (m,).

    Returns:
    np.ndarray: An array with softmax probabilities, shape (m,).
    """
    scaled_Z = Z - max(Z)
    m = Z.shape[0]
    ret = np.zeros(m)
    exp_sum = 0
    for i in range(m):
        ret[i] = np.exp(scaled_Z[i])
        exp_sum += np.exp(scaled_Z[i])
    ret = ret / exp_sum
    return ret


@typechecked
def softmax_backward(s: np.ndarray, loss_gradient: np.ndarray) -> np.ndarray:
    """
    Calculates the gradients of the softmax function.

    Parameters:
    s (np.ndarray): ...
    loss_gradient (np.ndarray): ...

    Returns:
    np.ndarray: ...
    """
    m = s.shape[0]
    jacobian = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                jacobian[i][j] = s[i] * (1 - s[i])
            else:
                jacobian[i][j] = -s[i] * s[j]
    gradients = np.dot(jacobian, loss_gradient)
    return gradients


@typechecked
def relu_forward(Z: np.ndarray) -> np.ndarray:
    """
    Apply the ReLU activation function to the input array.

    The ReLU (Rectified Linear Unit) function sets all negative values in the input array to zero.

    Parameters:
    Z (np.ndarray): The activations, shape (m,).

    Returns:
    np.ndarray: An array with all activation values below zero set to zero, shape (m,).
    """
    Z[Z < 0] = 0
    return Z


@typechecked
def relu_backward(Z: np.ndarray[float]) -> np.ndarray[float]:
    """
    Derivative of the ReLU function, see ReLU description in relu_forward.

    Parameters:
    Z (np.ndarray): An array to which apply the derivative of ReLU on, shape (m,).

    Returns:
    np.ndarray: An array with all values above zero equal to one, else zero.
    """
    Z[Z <= 0] = 0
    Z[Z > 0] = 1
    return Z
