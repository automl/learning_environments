import torch
import torch.nn as nn
import numpy as np
from time import time
from models.model_utils import build_nn_from_config
from torch.distributions.normal import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, agent_name, config):
        super().__init__()

        self.net = build_nn_from_config(input_dim=state_dim, output_dim=action_dim, nn_config=config["agents"][agent_name])
        self.action_std = torch.nn.Parameter(torch.ones(action_dim, device=device) * config["agents"][agent_name]["action_std"])

    def forward(self, state):
        action_mean = self.net(state)
        self.clamp(self.action_std, 0.01)
        dist = Normal(action_mean, self.action_std)
        return dist.rsample()

    def evaluate(self, state, action):
        action_mean = self.net(state)
        dist = Normal(action_mean, self.action_std)
        action_logprobs = torch.sum(dist.log_prob(action), dim=1)
        self.clamp(self.action_std, 0.01)
        dist_entropy = torch.sum(dist.entropy(), dim=1)
        return action_logprobs, dist_entropy

    def clamp(self, action_std, min_val):
        for i in range(len(action_std)):
            action_std.data[i] = max(action_std.data[i], min_val)

class Critic_Q(nn.Module):
    def __init__(self, state_dim, action_dim, agent_name, config):
        super().__init__()

        self.net = build_nn_from_config(input_dim=state_dim + action_dim, output_dim=action_dim, nn_config=config["agents"][agent_name])

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))


class Critic_V(nn.Module):
    def __init__(self, state_dim, agent_name, config):
        super().__init__()

        self.net = build_nn_from_config(input_dim=state_dim, output_dim=1, nn_config=config["agents"][agent_name])

    def forward(self, state):
        return self.net(state)
