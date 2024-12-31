import numpy as np
import gymnasium as gym
from typing import Callable
import ale_py
from enum import Enum
from collections import defaultdict
from typeguard import typechecked
from activation import relu_backward, relu_forward, softmax_backward, softmax_forward

gym.register_envs(ale_py)

NUM_NEURONS = 200
TRAIN_ITER = 2500
BATCH_SIZE = 10
LEARNING_RATE = 0.001


class ActionSpace(int, Enum):
    NOOP = 0
    FIRE = 1
    RIGHT = 2
    LEFT = 3
    RIGHTFIRE = 4
    LEFTFIRE = 5


@typechecked
def calculate_frame_diffs(start_frame: np.ndarray, end_frame: np.ndarray) -> np.ndarray:
    """
    Calculate the difference between two frames.

    parameters:
    start_frame (np.ndarray): The starting frame.
    end_frame (np.ndarray): The ending frame.

    Returns:
    np.ndarray: The normalized difference between the end frame and the start frame.
    """
    assert (
        start_frame.shape == end_frame.shape
    ), f"Shape mismatch, start frame shape: {start_frame.shape} vs end frame shape: {end_frame.shape}"
    return (end_frame - start_frame) / 256


@typechecked
def policy_gradient(
    x: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    env: gym.Env,
) -> np.ndarray:
    """
    Calculate the softmax probabilities using forward propagation.

    Given the inputs and weights/layers, this function performs forward propagation
    to calculate the softmax probabilities.

    Parameters:
    x (np.ndarray[float]): The input array, shape (128,).
    W1 (np.ndarray[np.ndarray[float]]): The first layer weights, shape (num_neurons, 128).
    W2 (np.ndarray[np.ndarray[float]]): The second layer weights, shape (6, num_neurons).
    env (gym.Env): The gym environment.

    Returns:
    np.ndarray[float]: The softmax probabilities, shape (6,).
    """

    # W1 shape: (num_neurons, 128), x shape: (128,)
    hidden_layer = np.dot(W1, x)
    hidden_layer = relu_forward(hidden_layer)

    # W2 shape: (6, num_neurons), hidden layer_shape: (num_neurons,)
    output = np.dot(W2, hidden_layer)
    probs = softmax_forward(output)
    return probs


def main() -> None:
    """
    Train a neural network policy to play Pong using policy gradient reinforcement learning.

    The network architecture consists of:
    - Input layer: 128 neurons (frame differences)
    - Hidden layer: 200 neurons with ReLU activation
    - Output layer: 6 neurons with softmax activation (action probabilities)

    Training runs for 20000 iterations, updating weights using policy gradients
    with cumulative rewards as scaling factors.
    """
    W1 = np.random.rand(NUM_NEURONS, 128)
    W2 = np.random.rand(6, NUM_NEURONS)

    for it in range(TRAIN_ITER):
        rewards: list[float] = []
        training_set: list[dict] = []

        for bt in range(BATCH_SIZE):
            env = gym.make("ALE/Pong-v5", render_mode=None, obs_type="ram")
            # env = gym.make("ALE/Pong-v5", render_mode="human", obs_type="ram")
            obs, _ = env.reset()
            prev_obs = obs
            cum_reward: int = 0

            done: bool = False
            while not done:
                obs, reward, term, trunc, info = env.step(ActionSpace.NOOP)
                cum_reward += reward
                done |= term or trunc

                frame_diff = calculate_frame_diffs(prev_obs, obs)
                action_probs = policy_gradient(x=frame_diff, env=env, W1=W1, W2=W2)
                action = np.random.choice(len(ActionSpace), p=action_probs, size=None)

                obs, reward, term, trunc, info = env.step(action)
                cum_reward += reward
                done |= term or trunc

                time_step = {"action": action, "probs": action_probs, "obs": frame_diff}
                training_set.append(time_step)
                prev_obs = obs
                # env.render()
            rewards.append(cum_reward)

        np_rewards = np.array(rewards)
        cum_reward = np.sum(rewards)
        for x in training_set:
            normalized_reward = (cum_reward - np.mean(np_rewards)) / (
                np.std(np_rewards) + 1e-10
            )
            true_val = [1 if i == x["action"] else 0 for i in ActionSpace]
            dJdz2 = x["probs"] - true_val

            dJda2 = softmax_backward(s=x["probs"], loss_gradient=dJdz2)  # shape is (6,)
            hidden_output = relu_forward(np.dot(W1, x["obs"]))
            dJdW2 = np.outer(dJda2, hidden_output)
            W2 -= LEARNING_RATE * normalized_reward * dJdW2

            dJdz1 = np.dot(W2.T, dJda2)  # shape is (NUM_NEURONS,)
            dJda1 = relu_backward(dJdz1)
            dJdW1 = np.outer(dJda1, x["obs"])
            W1 -= LEARNING_RATE * normalized_reward * dJdW1

        if it % 10 == 0:
            print(f"ITERATION {it} COMPLETED --- AVERAGE REWARD: {np.mean(np_rewards)}")
            with open(f"weights/w_{it}.npy", "wb") as f:
                np.save(f, W1)
                np.save(f, W2)

    env.close()


if __name__ == "__main__":
    main()
