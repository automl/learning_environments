import gym
import torch
import torch.nn as nn
import numpy as np
from utils import Identity


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VirtualEnv(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_seed = None
        self.env_name = str(kwargs["env_name"])
        self.state_dim = int(kwargs["state_dim"])
        self.action_dim = int(kwargs["action_dim"])
        self.zero_init = bool(kwargs["zero_init"])
        self.solved_reward = float(kwargs["solved_reward"])
        self.state = torch.zeros(self.state_dim, device=device)  # not sure what the default state should be

        # for compatibility
        self._max_episode_steps = int(kwargs["max_steps"])

        # for rendering
        if self.env_name != 'Test':
            self.viewer_env = gym.make(self.env_name)
            self.viewer_env.reset()

        hidden_size = int(kwargs["hidden_size"])
        weight_norm = bool(kwargs["weight_norm"])

        if weight_norm:
            weight_nrm = torch.nn.utils.weight_norm
        else:
            weight_nrm = Identity

        # self.base = nn.Sequential(
        #     weight_nrm(nn.Linear(self.state_dim + self.action_dim + 1, hidden_size)),  # +1 because of seed
        #     nn.PReLU(),
        #     weight_nrm(nn.Linear(hidden_size, hidden_size)),
        #     nn.PReLU(),
        #     weight_nrm(nn.Linear(hidden_size, hidden_size)),
        #     nn.PReLU(),
        # ).to(device)
        # self.state_head = weight_nrm(nn.Linear(hidden_size, self.state_dim)).to(device)
        # self.reward_head = weight_nrm(nn.Linear(hidden_size, 1)).to(device)
        # self.done_head = weight_nrm(nn.Linear(hidden_size, 1)).to(device)

        self.state_net = nn.Sequential(
            weight_nrm(nn.Linear(self.state_dim + self.action_dim + 1, hidden_size)),  # +1 because of seed
            nn.PReLU(),
            weight_nrm(nn.Linear(hidden_size, hidden_size)),
            nn.PReLU(),
            weight_nrm(nn.Linear(hidden_size, hidden_size)),
            nn.PReLU(),
            weight_nrm(nn.Linear(hidden_size, self.state_dim))
        ).to(device)

        self.reward_net = nn.Sequential(
            weight_nrm(nn.Linear(self.state_dim + self.action_dim + 1, hidden_size)),  # +1 because of seed
            nn.PReLU(),
            weight_nrm(nn.Linear(hidden_size, hidden_size)),
            nn.PReLU(),
            weight_nrm(nn.Linear(hidden_size, hidden_size)),
            nn.PReLU(),
            weight_nrm(nn.Linear(hidden_size, 1))
        ).to(device)

        self.done_net = nn.Sequential(
            weight_nrm(nn.Linear(self.state_dim + self.action_dim + 1, hidden_size)),  # +1 because of seed
            nn.PReLU(),
            weight_nrm(nn.Linear(hidden_size, hidden_size)),
            nn.PReLU(),
            weight_nrm(nn.Linear(hidden_size, hidden_size)),
            nn.PReLU(),
            weight_nrm(nn.Linear(hidden_size, 1))
        ).to(device)

    def reset(self):
        if self.zero_init:
            self.state = torch.zeros(self.state_dim, device=device)
        elif self.env_name == "Pendulum-v0" or self.env_name == "Dummy":
            self.state = torch.tensor(
                [np.random.uniform(low=-1, high=1), np.random.uniform(low=-1, high=1), np.random.uniform(low=-5, high=5)],
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

    def get_state(self):
        return self.state

    def step(self, action, state, input_seed=0):
        input = torch.cat((action, state, input_seed), dim=len(action.shape) - 1)
        next_state = self.state_net(input)
        reward = self.reward_net(input)
        done = (self.done_net(input) > 0.5).float()
        return next_state, reward, done

        # input = torch.cat((action, state, input_seed), dim=len(action.shape) - 1)
        # x = self.base(input)
        # next_state = self.state_head(x)
        # reward = self.reward_head(x)
        # done = (self.done_head(x) > 0.5).float()
        # self.state = next_state
        # return next_state, reward, done

    def render(self):
        if self.env_name != 'Test':
            self.viewer_env.env.state = self.state.cpu().data.numpy()
            self.viewer_env.render()

    def close(self):
        if self.env_name != 'Test':
            if self.viewer_env.viewer:
                self.viewer_env.viewer.close()
                self.viewer_env.viewer = None
