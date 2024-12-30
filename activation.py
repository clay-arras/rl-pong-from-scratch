import numpy as np


def softmax_forward(Z: np.ndarray[float]) -> np.ndarray[float]:
    """Desc: gives an array of softmax probabilities between 0 and 1"""
    m = Z.shape[0]
    ret = np.zeros(m)
    exp_sum = 0
    for i in range(m):
        ret[i] = np.exp(Z[i])
        exp_sum += np.exp(Z[i])
    ret = ret / exp_sum
    return ret


def relu_forward(Z: np.ndarray[float]) -> np.ndarray[float]:
    """Desc: rectified linear unit, sets all indices with values below zero to zero"""
    Z[Z < 0] = 0
    return Z


def softmax_backward(
    s: np.ndarray[float], loss_gradient: np.ndarray[float]
) -> np.ndarray[float]:
    """Desc: calculates the gradients of the softmax function"""
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


def relu_backward(Z: np.ndarray[float]) -> np.ndarray[float]:
    Z[Z <= 0] = 0
    Z[Z > 0] = 1
    return Z

