import gymnasium as gym
import numpy as np
from enum import Enum


class ActionSpace(int, Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def main() -> None:
    env = gym.make("FrozenLake-v1", render_mode="human")
    obs, _ = env.reset()

    done: False = False
    while not done: 
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        done |= term or trunc
        
        env.render()


if __name__ == "__main__":
    main()