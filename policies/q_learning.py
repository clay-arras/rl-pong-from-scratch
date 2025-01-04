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


def in_bounds(x: np.ndarray[int], y: np.ndarray[int]) -> np.ndarray[bool]:
    return np.logical_and(
        np.logical_and(0 <= x, x < N_ROWS),
        np.logical_and(0 <= y, y < N_COLS)
    )


def epsilon_greedy(q_table: np.ndarray, pos: np.ndarray[tuple[int, int]], epsilon: float = 0.05) -> ActionSpace:
    """
    Select an action using the epsilon-greedy policy.

    Parameters:
    q_table (np.ndarray): The Q-table containing Q-values for each state-action pair.
    pos (tuple[int, int]): The current position in the environment, shape (num_envs, 2)
    epsilon (float): The probability of selecting a random action (exploration rate).

    Returns:
    ActionSpace: The selected action, range 0-3
    """
    num_envs = len(pos.T)
    # Directions is (4,2)
    if np.random.uniform(0, 1) < epsilon:
        pos_arr = np.array([DIRECTIONS + p for p in pos.T])
        for i, v in enumerate(pos_arr):
            pos_arr[i] = np.array([p for p in v if in_bounds(p[0], p[1])])
        ret = np.zeros(num_envs)
        for i in range(num_envs):
            ret[i] = np.random.choice(pos_arr[i], size=None)
        return ret

    best_action: ActionSpace = 0
    best_value: float = float('-inf')

    # (num_envs, 4, 2)
    pos_arr = [DIRECTIONS + p for p in pos.T]
    # (num_envs, 4, 2)
    for i, v in enumerate(pos_arr):
        pos_arr[i] = [p for p in v if in_bounds(p[0], p[1])]
    action_q = [[] for i in range(len(pos_arr))]
    for i, v in enumerate(pos_arr):
        action_q[i] = [q_table[i[0]][i[1]] for i in v]

    best_action = np.zeros(num_envs)
    best_value = np.zeros(num_envs)
    for i in ActionSpace:
        new_pos: np.ndarray = pos.T + DIRECTIONS[i]
        for j, v in enumerate(new_pos):
            if in_bounds(v[0], v[1]):
                q_value = q_table[v[0]][v[1]]
                if q_value > best_value[j]:
                    best_value[j] = q_value
                    best_action[j] = i
                elif q_value == best_value[j] and np.random.random() < 0.5:
                    best_action[j] = i

    for i in range(len(action_q)):
        if len(set(action_q[i])) == 1:
            best_action[i] = np.random.choice(len(ActionSpace), size=None)

    return best_action


def main() -> None:
    env = gym.make_vec(
        "FrozenLake-v1",
        num_envs=3,
        vectorization_mode="sync",
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
            action = epsilon_greedy(q_table=q_table, pos=np.array([curr_x, curr_y]))
            action = np.array([int(i) for i in action])

            new_obs, reward, term, trunc, _ = env.step(action)
            next_x, next_y = divmod(new_obs, N_COLS)

            # Bellman update equation (TD update)
            # next_max_q = np.max([
            #     q_table[next_x + i[0]][next_y + i[1]]
            #     for i in DIRECTIONS
            #     if in_bounds(next_x + i[0], next_y + i[1])
            # ])
            get_max_q = lambda x: np.max([q_table[i[0]][i[1]] for i in x])
            next_max_q = np.vectorize(get_max_q)(DIRECTIONS[in_bounds(next_x + DIRECTIONS[0], next_y + DIRECTIONS[1])])
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
