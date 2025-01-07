import gymnasium as gym
import numpy as np
from enum import Enum
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

ALPHA = 0.2
GAMMA = 0.8
NUM_ITER = 1000
N_COLS = 4
N_ROWS = 4
NUM_ENVS = 10
DIRS = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)])


class ActionSpace(int, Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def in_bounds(x: np.ndarray[int], y: np.ndarray[int]) -> np.ndarray[bool]:
    """Check if coordinates are within environment bounds.

    Parameters:
    x (np.ndarray[int]): Array of x coordinates, shape (num_envs,)
    y (np.ndarray[int]): Array of y coordinates, shape (num_envs,)

    Returns:
    np.ndarray[bool]: Boolean mask indicating which coordinates are valid
    """
    return np.logical_and(
        np.logical_and(0 <= x, x < N_ROWS), np.logical_and(0 <= y, y < N_COLS)
    )


def get_possible_q_values(q_table: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Get Q-values for all possible actions from current positions.

    For each environment, gets the Q-values of valid adjacent positions from the Q-table.
    Invalid positions (out of bounds) are assigned -inf Q-values.

    Parameters:
    q_table (np.ndarray): Q-value table of shape (n_rows, n_cols) containing values for each position
    pos (np.ndarray): Current positions array of shape (num_envs, 2) containing (x,y) coordinates

    Returns:
    np.ndarray: Array of shape (num_envs, 4) containing Q-values for each possible action.
                Invalid actions are assigned -inf.
    """
    num_envs = pos.shape[0]

    directions_expanded = np.expand_dims(DIRS, axis=0)
    positions_expanded = np.expand_dims(pos, axis=1)
    adjacent_positions = (
        directions_expanded + positions_expanded
    )  # shape (num_envs, 4, 2)
    positions_flattened = adjacent_positions.reshape(
        4 * num_envs, 2
    ).T  # shape (2, 4*num_envs)
    valid_positions_mask = in_bounds(
        x=positions_flattened[0], y=positions_flattened[1]
    )  # shape (4*num_envs,)
    valid_actions_mask = valid_positions_mask.reshape(
        num_envs, 4
    )  # shape (num_envs, 4)

    q_values = np.full((num_envs, 4), -np.inf)
    valid_x_coords = adjacent_positions[..., 0][valid_actions_mask]
    valid_y_coords = adjacent_positions[..., 1][valid_actions_mask]
    q_values[valid_actions_mask] = q_table[valid_x_coords, valid_y_coords]
    return q_values


def epsilon_greedy(
    q_table: np.ndarray, pos: np.ndarray, epsilon: float = 0.05
) -> ActionSpace:
    """
    Select an action using the epsilon-greedy policy.

    For all valid adjacent spaces, choose the one with the highest Q-value.
    If multiple spaces have the same highest Q-value, randomly select one of them.

    Parameters:
    q_table (np.ndarray): The Q-table containing Q-values for each state-action pair.
    pos (tuple[int, int]): The current position in the environment, shape (num_envs, 2)
    epsilon (float): The probability of selecting a random action (exploration rate).

    Returns:
    ActionSpace: The selected action, range 0-3
    """
    q_values = get_possible_q_values(q_table=q_table, pos=pos)
    valid_actions_mask = ~np.equal(q_values, np.full(q_values.shape, -np.inf));

    rng = np.random.default_rng()
    if np.random.uniform(0, 1) < epsilon:
        selected_actions = np.argmax(
            valid_actions_mask * rng.random(valid_actions_mask.shape), axis=1
        )
        return selected_actions

    best_q_values = np.max(q_values, axis=1)
    best_actions_mask = np.equal(q_values, best_q_values[:, np.newaxis])
    selected_actions = np.argmax(
        best_actions_mask * rng.random(best_actions_mask.shape), axis=1
    )

    return selected_actions


def main() -> None:
    env = gym.make_vec(
        "FrozenLake-v1",
        num_envs=NUM_ENVS,
        vectorization_mode="sync",
        is_slippery=False,
        render_mode="human",
        desc=generate_random_map(size=N_ROWS),
    )
    q_table = np.zeros((N_ROWS, N_COLS))

    for _ in range(NUM_ITER):
        obs, _ = env.reset()
        done: bool = False
        discount: int = 0.9
        while not done:
            curr_x, curr_y = divmod(obs, N_ROWS)
            action = epsilon_greedy(q_table=q_table, pos=np.array([curr_x, curr_y]).T)
            obs, reward, term, trunc, _ = env.step(action)
            next_x, next_y = divmod(obs, N_ROWS)

            # Bellman update equation (TD update)
            q_values = get_possible_q_values(q_table=q_table, pos=np.array([next_x, next_y]).T)
            next_q_values = np.max(q_values, axis=1)
            q_table[curr_x, curr_y] += ALPHA * (
                reward + discount * next_q_values - q_table[curr_x, curr_y]
            )

            discount *= GAMMA
            has_autoreset = np.logical_or(term, trunc)
            done = done or (has_autoreset.sum() > NUM_ENVS/2)
            env.render()

        print(q_table)


if __name__ == "__main__":
    main()
