import gym
from gym.utils import seeding
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from os import path
from copy import deepcopy


class TestEnv():
    def __init__(self):
        high = np.array([1,1])
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def seed(self, seed=None):
        pass

    def step(self, u):
        self.state = self.state + np.sin(u)
        return self.state, math.cos(u), False, {}

    def reset(self):
        high = np.array([np.pi, np.pi])
        np_random, _ = seeding.np_random()
        state_tmp = np_random.uniform(low=-high, high=high)
        self.state = np.array([np.cos(state_tmp[0]), np.sin(state_tmp[0])])
        return self.state

    def render(self, mode="human"):
        pass

    def close(self):
        pass

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
