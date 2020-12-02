import os
import numpy as np
import torch
import torch.nn as nn
from envs.virtual_env import VirtualEnv
from gym.spaces import Discrete
from utils import to_one_hot_encoding, from_one_hot_encoding

class EnvWrapper(nn.Module):
    # wraps a gym/virtual environment for easier handling
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.same_action_num = None

    def step(self, action, state=None):
        same_action_num = 1

        if self.is_virtual_env():
            reward_sum = None

            if self.has_discrete_action_space():
                action = to_one_hot_encoding(action, self.get_action_dim())

            if state is None:
                for i in range(same_action_num):
                    state, reward, done = self.env.step(action=action.to(self.env.device))
                    if reward_sum is None:
                        reward_sum = reward
                    else:
                        reward_sum += reward
                    # TODO: proper handling of the done flag for a batch of states/actions if same_action_num > 1
            else:
                for i in range(same_action_num):
                    state, reward, done = self.env.step(action=action.to(self.env.device), state=state.to(self.env.device))
                    if reward_sum is None:
                        reward_sum = reward
                    else:
                        reward_sum += reward
                    # TODO: proper handling of the done flag for a batch of states/actions if same_action_num > 1

            reward = reward_sum.to("cpu")
            next_state = state.to("cpu")
            done = done.to("cpu")

            if self.has_discrete_state_space():
                next_state = from_one_hot_encoding(next_state)

            return next_state, reward, done

        else:
            action = action.cpu().detach().numpy()
            if self.has_discrete_action_space():
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
        state = self.env.reset()

        if type(state) == np.ndarray:
            state_torch = torch.from_numpy(state).float().cpu()
        elif torch.is_tensor(state):
            state_torch = state
        else:  # float
            state_torch = torch.tensor([state], device="cpu", dtype=torch.float32)

        if self.has_discrete_state_space() and self.is_virtual_env():
            return from_one_hot_encoding(state_torch)
        else:
            return state_torch

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
            return self.env.observation_space.n

    def get_action_dim(self):
        if self.env.action_space.shape:
            return self.env.action_space.shape[0]
        else:
            return self.env.action_space.n

    def get_max_action(self):
        if self.env.env_name == 'Pendulum-v0':
            return 2
        else:
            return 1

    def has_discrete_action_space(self):
        if isinstance(self.env.action_space, Discrete):
            return True
        else:
            return False

    def has_discrete_state_space(self):
        if isinstance(self.env.observation_space, Discrete):
            return True
        else:
            return False

    def render(self):
        if not self.is_virtual_env():
            return self.env.render()

    def close(self):
        if not self.is_virtual_env():
            return self.env.close()

    def get_solved_reward(self):
        return self.env.solved_reward

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

    def set_agent_params(self, same_action_num, gamma):
        self.same_action_num = same_action_num
        if hasattr(self.env, "set_agent_params"):
            self.env.set_agent_params(gamma=gamma)



