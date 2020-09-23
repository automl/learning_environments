import os
import numpy as np
import torch
import torch.nn as nn
from envs.virtual_env import VirtualEnv
from gym.spaces import Discrete


class EnvWrapper(nn.Module):
    # wraps a gym/virtual environment for easier handling
    def __init__(self, env):
        super().__init__()
        self.env = env

    def step(self, action, same_action_num=1):
        if self.is_virtual_env():
            reward_sum = None

            for i in range(same_action_num):
                state, reward, done = self.env.step(action.to(self.env.device))
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

            if next_state_torch.dim() == 0:
                next_state_torch = next_state_torch.unsqueeze(0)

            return next_state_torch, reward_torch, done_torch

    def reset(self):
        val = self.env.reset()
        if type(val) == np.ndarray:
            return torch.from_numpy(val).float().cpu()
        elif torch.is_tensor(val):
            return val
        else:   # torch
            return torch.tensor([val], device="cpu", dtype=torch.float32)

    def get_random_action(self):
        action = self.env.action_space.sample()
        if type(action) == np.ndarray:
            return torch.from_numpy(self.env.action_space.sample())
        elif type(action) == int:
            return torch.tensor([action], device="cpu", dtype=torch.float32)

    def get_state_dim(self):
        if self.env.observation_space.shape:
            return self.env.observation_space.shape[0]
        else:
            #return self.env.observation_space.n
            return 1

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
            print("Unknownn environment, performance may decrease")
            return 0

    def is_discrete_action_space(self):
        if isinstance(self.env.action_space, Discrete):
            return True
        else:
            return False

    def render(self):
        if not self.is_virtual_env():
            return self.env.render()

    def close(self):
        if not self.is_virtual_env():
            return self.env.close()

    def max_episode_steps(self):
        return self.env._max_episode_steps

    def seed(self, seed):
        if not self.is_virtual_env():
            return self.env.seed(seed)
        else:
            print("Setting manuel seed not yet implemented, performance may decrease")
            return 0

    def is_virtual_env(self):
        return isinstance(self.env, VirtualEnv)


