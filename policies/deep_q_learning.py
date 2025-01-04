import gymnasium as gym
import numpy as np
from enum import Enum
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

ALPHA = 0.2
GAMMA = 0.8
NUM_ITER = 1000
N_COLS = 8
N_ROWS = 8
DIRECTIONS = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)])


class ActionSpace(int, Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < N_ROWS and 0 <= y < N_COLS


def epsilon_greedy(q_table: np.ndarray, pos: tuple[int, int], epsilon: float = 0.05) -> ActionSpace:
    """
    Select an action using the epsilon-greedy policy.

    Parameters:
    q_table (np.ndarray): The Q-table containing Q-values for each state-action pair.
    pos (tuple[int, int]): The current position in the environment.
    epsilon (float): The probability of selecting a random action (exploration rate).

    Returns:
    ActionSpace: The selected action, range 0-3
    """
    if np.random.uniform(0, 1) < epsilon:
        pos_arr = DIRECTIONS + pos
        pos_arr = [i for i, p in enumerate(pos_arr) if in_bounds(p[0], p[1])]
        return np.random.choice(pos_arr, size=None)

    best_action: ActionSpace = 0
    best_value: float = float('-inf')

    pos_arr = DIRECTIONS + pos
    pos_arr = [p for p in pos_arr if in_bounds(p[0], p[1])]
    action_q = [q_table[i[0]][i[1]] for i in pos_arr]

    if len(set(action_q)) == 1:
        return np.random.choice(len(ActionSpace), size=None)

    for i in ActionSpace:
        new_pos = pos + DIRECTIONS[i]
        if in_bounds(new_pos[0], new_pos[1]):
            q_value = q_table[new_pos[0]][new_pos[1]]
            if q_value > best_value:
                best_value = q_value
                best_action = i
            elif q_value == best_value and np.random.random() < 0.5:
                best_action = i

    return best_action


def main() -> None:
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False,
        render_mode="human",
        desc=generate_random_map(size=8),
    )
    q_table = np.zeros((N_ROWS, N_COLS))

    for _ in range(NUM_ITER):
        obs, _ = env.reset()
        done: bool = False
        discount = 1
        while not done:
            curr_x, curr_y = divmod(obs, N_COLS)
            action = epsilon_greedy(q_table=q_table, pos=(curr_x, curr_y))

            new_obs, reward, term, trunc, _ = env.step(action)
            next_x, next_y = divmod(new_obs, N_COLS)

            # Bellman update equation (TD update)
            next_max_q = np.max([
                q_table[next_x + i[0]][next_y + i[1]]
                for i in DIRECTIONS
                if in_bounds(next_x + i[0], next_y + i[1])
            ])
            q_table[curr_x][curr_y] += ALPHA * (
                reward + discount * next_max_q - q_table[curr_x][curr_y]
            )

            discount *= GAMMA
            obs = new_obs
            done |= term or trunc
            env.render()
        print(q_table)


if __name__ == "__main__":
    main()
