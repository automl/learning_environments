import gym
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
        self.solved_reward = kwargs['solved_reward']
        self.state = torch.zeros(self.state_dim, device=device)   # not sure what the default state should be

        # for compatibility
        self._max_episode_steps = int(kwargs['max_steps'])

        # for rendering
        self.viewer_env = gym.make(self.env_name)
        self.viewer_env.reset()

        hidden_size = int(kwargs["hidden_size"])
        self.base = nn.Sequential(
            weight_norm(nn.Linear(self.state_dim + self.action_dim + 1, hidden_size)),  # +1 because of seed
            nn.ReLU(),
            weight_norm(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        ).to(device)
        self.state_head = weight_norm(nn.Linear(hidden_size, self.state_dim)).to(device)
        self.reward_head = weight_norm(nn.Linear(hidden_size, 1)).to(device)
        self.done_head = weight_norm(nn.Linear(hidden_size, 1)).to(device)

    def reset(self):
        if self.zero_init:
            self.state = torch.zeros(self.state_dim, device=device)
        elif self.env_name == "Pendulum-v0":
            self.state = torch.tensor([
                    np.random.uniform(low=-np.pi, high=np.pi),
                    np.random.uniform(low=-1, high=1),
                    np.random.uniform(low=-8, high=8),
                ],
                device=device,
                dtype=torch.float32,
            )
        else:
            raise NotImplementedError("Unknown environment: non-zero state reset only for supported environments.")

        return self.state

    def set_input_seed(self, input_seed):
        self.input_seed = input_seed

    def get_input_seed(self):
        return self.input_seed

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def step(self, action, state, input_seed=0):
        input = torch.cat((action, state, input_seed), dim=len(action.shape)-1)
        x = self.base(input)
        next_state = self.state_head(x)
        reward = self.reward_head(x)
        done = (self.done_head(x) > 0.5).float()
        self.state = next_state
        return next_state, reward, done

    def render(self):
        self.viewer_env.env.state = self.state.cpu().data.numpy()
        self.viewer_env.render()

    def close(self):
        if self.viewer_env.viewer:
            self.viewer_env.viewer.close()
            self.viewer_env.viewer = None
