from q_learning import epsilon_greedy
import numpy as np


def test_epsilon_greedy() -> None:
    q_table = np.array(
        [
            [1, 2, 3, 4],
            [5, 3, 1, 2],
            [3, 2, 0, 2],
            [2, 2, 0, 1],
        ]
    )
    pos = np.array(
        [
            [1, 2],
            [3, 4],
            [2, 0],
        ]
    )
    res = epsilon_greedy(q_table=q_table, pos=pos)
