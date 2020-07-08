import os
import torch
import torch.nn as nn
from envs.virtual_env import VirtualEnv


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EnvWrapper(nn.Module):
    # wraps a gym/virtual environment for easier handling
    def __init__(self, env):
        super().__init__()
        self.env = env

    def step(self, action, state, input_seed=torch.tensor([0], device="cpu", dtype=torch.float32)):
        if self.is_virtual_env():
            next_state, reward, done = self.env.step(action.to(device), state.to(device), input_seed.to(device))
            reward = reward.to("cpu")
            next_state = next_state.to("cpu")
            done = done.to("cpu")
            return next_state, reward, done
        else:
            self.env.state = state.cpu().detach().numpy()
            next_state, reward, done, _ = self.env.step(action.cpu().detach().numpy())
            next_state_torch = torch.tensor(next_state, device="cpu", dtype=torch.float32)
            reward_torch = torch.tensor(reward, device="cpu", dtype=torch.float32)
            done_torch = torch.tensor(done, device="cpu", dtype=torch.float32)
            return next_state_torch, reward_torch, done_torch

    def reset(self):
        if self.is_virtual_env():
            return self.env.reset()
        else:
            return torch.from_numpy(self.env.reset()).float().cpu()

    def get_random_action(self):
        # do random action in the [-1,1] range
        return torch.empty(self.get_action_dim(), device="cpu", dtype=torch.float32).uniform_(-1, 1)

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

    def render(self, state):
        if self.is_virtual_env():
            return self.env.render(state)
        else:
            self.env.state = state.cpu().detach().numpy()
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

    def get_state_dict(self):
        if self.is_virtual_env():
            return self.env.get_state_dict()
        else:
            return self.kwargs

    def set_state_dict(self, env_state):
        if self.is_virtual_env():
            self.env.set_state_dict(env_state)
        else:
            for key, value in env_state.items():
                setattr(self.env, key, value)

    def save(self, path):
        torch.save(self.get_state_dict(), path)

    def load(self, path):
        if os.path.isfile(path):
            self.set_state_dict(torch.load(path))
        else:
            raise FileNotFoundError('File not found: ' + str(path))
