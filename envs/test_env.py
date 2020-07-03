import gym
import gym.envs.classic_control
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from copy import deepcopy


class TestEnv():
    def __init__(self):
        high = np.array([1,1])
        self.action_space = spaces.Box(low=0, high=2*np.pi, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def seed(self, seed=None):
        pass

    def step(self, u):
        return np.concatenate((np.sin(u), np.sin(u)), axis=0), 0, False, {}

    def reset(self):
        return np.array([0])

    def render(self, mode="human"):
        pass

    def close(self):
        pass

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
