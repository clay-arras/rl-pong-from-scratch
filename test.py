import numpy as np
import gymnasium as gym
from typing import Callable
import ale_py
from enum import Enum
from collections import defaultdict

gym.register_envs(ale_py)


class ActionSpace(Enum):
    NOOP = 0
    FIRE = 1
    RIGHT = 2
    LEFT = 3
    RIGHTFIRE = 4
    LEFTFIRE = 5


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


def relu_foward(Z: np.ndarray[float]) -> np.ndarray[float]:
    """Desc: rectified linear unit, sets all indices with values below zero to zero"""
    Z[Z < 0] = 0
    return Z


def softmax_backward(Z: np.ndarray[float]) -> np.ndarray[float]:
    """Desc: calculates the gradients of the softmax function"""
    s = softmax_forward(Z)
    m = s.shape[0]
    jacobian = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                jacobian[i][j] = s[i] * (1 - s[i])
            else: 
                jacobian[i][j] = -s[i] * s[j]
    pass


def forward_propagation(
    W: np.ndarray[np.ndarray[float]],
    X: np.ndarray[float],
    activation_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    # """Desc: calculates forward propogation for a single neuron layer"""
    # Z = np.dot(W, X)
    # return activation_fn(Z)
    pass


def backward_propogation() -> None:
    pass


def calculate_frame_diffs(
    start_frame: np.ndarray[float], end_frame: np.ndarray[float]
) -> np.ndarray[float]:
    # assert(start_frame.shape == (210, 160, 3))
    # assert(end_frame.shape == (210, 160, 3))
    return end_frame - start_frame


def policy_gradient(
    x: np.ndarray[float],
    W1: np.ndarray[np.ndarray[float]],
    W2: np.ndarray[np.ndarray[float]],
    env: gym.Env,
) -> ActionSpace:
    """Desc: Given the inputs and weights/layers, we will calculate the softmax probabilities using forward propogation"""

    # W1 shape: (num_neurons, 128), x shape: (128,)
    hidden_layer = np.dot(W1, x)

    # hidden_layer shape: (num_neurons,)
    hidden_layer = relu(hidden_layer)

    # W2 shape: (6, num_neurons), hidden layer_shape: (num_neurons,)
    output = np.dot(W2, hidden_layer)

    # output: (6,)
    probs = softmax(output)

    return probs.sample()


def main() -> None:
    env = gym.make("ALE/Pong-v5", render_mode="human", obs_type="ram")
    obs = env.reset()

    done = False
    cum_reward = 0

    training_set: list[dict] = []
    while not done:
        obs, reward, term, trunc, info = env.step(ActionSpace.NOOP)
        cum_reward += reward
        done |= term or trunc

        frame_diff = calculate_frame_diffs(prev_obs, obs)
        action = policy_gradient(x=frame_diff, env=env)
        obs, reward, term, trunc, info = env.step(action)

        done |= term or trunc

        time_step = defaultdict(lambda x: [])
        time_step["action"] = action
        time_step["obs"] = frame_diff
        training_set.append(time_step)

        prev_obs = obs
        env.render()

    for x in training_set:
        x["action"] # shape: (6,)
        x["obs"] # shape: (128,)

    # result = (cum_reward > 0)
    # update all the subsequent updates to cum_reward * gradient
    env.close()


if __name__ == "__main__":
    main()
