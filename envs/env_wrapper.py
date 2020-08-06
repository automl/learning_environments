import os
import torch
import torch.nn as nn
import numpy as np
from envs.virtual_env import VirtualEnv
from gym.envs.mujoco import mujoco_env
from gym.utils import seeding


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EnvWrapper(nn.Module):
    # wraps a gym/virtual environment for easier handling
    def __init__(self, env):
        super().__init__()
        self.env = env

    def step(self, action, state, input_seed=torch.tensor([0], device="cpu", dtype=torch.float32), same_action_num=1):
        if self.is_virtual_env():
            reward_sum = None

            for i in range(same_action_num):
                state, reward, done = self.env.step(action.to(device), state.to(device), input_seed.to(device))
                if reward_sum is None:
                    reward_sum = reward
                else:
                    reward_sum += reward
                # TODO: proper handling of the done flag for a batch of states/actions if same_action_num > 1

            reward = reward_sum.to("cpu")
            next_state = state.to("cpu")
            done = done.to("cpu")
            return next_state, reward, done

        else:
            action = action.cpu().detach().numpy()
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
        if self.is_virtual_env():
            return self.env.reset()
        else:
            return torch.from_numpy(self.env.reset()).float().cpu()

    def get_random_action(self):
        if self.is_virtual_env():
            ma = self.get_max_action()
            return torch.empty(self.get_action_dim(), device="cpu", dtype=torch.float32).uniform_(-ma, ma)
        else:
            return torch.from_numpy(self.env.action_space.sample())

    def get_state_dim(self):
        if self.is_virtual_env():
            return self.env.state_dim
        else:
            return self.env.observation_space.shape[0]

    def get_action_dim(self):
        if self.is_virtual_env():
            return self.env.action_dim
        else:
            return self.env.action_space.shape[0]

    def get_max_action(self):
        if self.env.env_name == 'Pendulum-v0':
            return 2
        elif self.env.env_name == 'MountainCarContinuous-v0':
            return 1
        elif self.env.env_name == "HalfCheetah-v2":
            return 1
        else:
            raise NotImplementedError("Unknownn RL agent")

    def render(self, state, action):
        if self.is_virtual_env():
            return self.env.render(state, action)
        else:
            return self.env.render()

    def close(self):
        return self.env.close()

    def max_episode_steps(self):
        return self.env._max_episode_steps

    def seed(self, seed):
        if isinstance(self.env, VirtualEnv):
            print("Virtual environment, no need to set a seed for random numbers")
        else:
            return self.env.seed(seed)

    def is_virtual_env(self):
        return isinstance(self.env, VirtualEnv)

    def is_mujoco_env(self):
        return isinstance(self.env, mujoco_env.MujocoEnv)

