import os
import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Discrete


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EnvWrapper(nn.Module):
    # wraps a gym/virtual environment for easier handling
    def __init__(self, env):
        super().__init__()
        self.env = env

    def step(self, action, state, same_action_num=1):
        action = action.cpu().detach().numpy()
        if self.is_discrete_action_space():
            action = action.astype(int)[0]

        reward_sum = 0
        for i in range(same_action_num):
            state, reward, done, _ = self.env.step(action)
            reward_sum += reward
            if done:
                break

        next_state_torch = torch.tensor(state, device="cpu", dtype=torch.float32)
        reward_torch = torch.tensor(reward_sum, device="cpu", dtype=torch.float32)
        done_torch = torch.tensor(done, device="cpu", dtype=torch.float32)
        return next_state_torch, reward_torch, done_torch

    def reset(self):
        return torch.from_numpy(self.env.reset()).float().cpu()

    def get_random_action(self):
        action = self.env.action_space.sample()
        if type(action) == np.ndarray:
            return torch.from_numpy(self.env.action_space.sample())
        elif type(action) == int:
            return torch.tensor([action], device="cpu", dtype=torch.float32)

    def get_state_dim(self):
        return self.env.observation_space.shape[0]

    def get_action_dim(self):
        if self.env.action_space.shape:
            return self.env.action_space.shape[0]
        else:
            return self.env.action_space.n

    def get_max_action(self):
        if self.env.env_name == 'Pendulum-v0':
            return 2
        elif self.env.env_name == 'MountainCarContinuous-v0':
            return 1
        elif self.env.env_name == "HalfCheetah-v2":
            return 1
        else:
            raise NotImplementedError("Unknownn RL agent")

    def is_discrete_action_space(self):
        if isinstance(self.env.action_space, Discrete):
            return True
        else:
            return False

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def max_episode_steps(self):
        return self.env._max_episode_steps

    def seed(self, seed):
        return self.env.seed(seed)


