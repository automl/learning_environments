import gym
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VirtualEnv(nn.Module, gym.Env):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(VirtualEnv, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_seed = None
        self.state = torch.zeros(state_dim, device="cpu")  # not sure what the default state should be
        self._max_episode_steps = int(kwargs["max_steps"])

        hidden_size = kwargs["hidden_size"]
        self.base = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_size),  # +1 because of seed
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.state_head = nn.Linear(hidden_size, state_dim)
        self.reward_head = nn.Linear(hidden_size, 1)
        self.done_head = nn.Linear(hidden_size, 1)

    def reset(self):
        self.state = torch.zeros(self.state_dim, device="cpu")  # not sure what the default state should be
        return self.state

    def set_seed(self, seed):
        self.input_seed = seed

    def step(self, action):
        if not self.input_seed:
            raise ValueError("no input seed set")

        input = torch.cat((action, self.state, self.input_seed))
        x = self.base(input)
        next_state = self.state_head(x)
        reward = self.reward_head(x)
        done = self.done_head(x) > 0.5
        return next_state, reward, done, {}
