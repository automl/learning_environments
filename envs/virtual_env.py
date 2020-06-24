import gym
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.weight_norm import weight_norm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VirtualEnv(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_seed = None
        self.env_name = kwargs['env_name']
        self.state_dim = int(kwargs['state_dim'])
        self.action_dim = int(kwargs['action_dim'])
        self.zero_init = bool(kwargs['zero_init'])
        self.state = torch.zeros(self.state_dim, device='cpu')   # not sure what the default state should be

        # for compatibility
        self._max_episode_steps = int(kwargs['max_steps'])
        # for rendering
        self.viewer_env = gym.make(self.env_name)
        self.viewer_env.reset()

        hidden_size = int(kwargs["hidden_size"])
        self.base = nn.Sequential(
            weight_norm(nn.Linear(self.state_dim + self.action_dim + 1, hidden_size)),  # +1 because of seed
            nn.Tanh(),
            weight_norm(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )
        self.state_head = nn.utils.weight_norm(nn.Linear(hidden_size, self.state_dim))
        self.reward_head = nn.utils.weight_norm(nn.Linear(hidden_size, 1))
        self.done_head = nn.utils.weight_norm(nn.Linear(hidden_size, 1))

    def reset(self):
        if self.zero_init:
            self.state = torch.zeros(self.state_dim, device="cpu")
        elif self.env_name == "Pendulum-v0":
            self.state = torch.tensor([
                    np.random.uniform(low=-np.pi, high=np.pi),
                    np.random.uniform(low=-1, high=1),
                    np.random.uniform(low=-8, high=8),
                ],
                device="cpu",
                dtype=torch.float32,
            )
        else:
            raise NotImplementedError("Unknown environment: non-zero state reset only for supported environments.")

        return self.state

    def set_seed(self, seed):
        self.input_seed = seed

    def step(self, action):
        input = torch.cat((action, self.state, self.input_seed))
        x = self.base(input)
        next_state = self.state_head(x)
        reward = self.reward_head(x)
        done = self.done_head(x) > 0.5
        self.state = next_state
        return next_state, reward, done

    def render(self):
        self.viewer_env.env.state = self.state.data.numpy()
        self.viewer_env.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
