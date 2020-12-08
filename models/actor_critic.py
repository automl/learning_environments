import copy

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from models.model_utils import build_nn_from_config


class Actor_TD3(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, agent_name, config):
        super().__init__()

        self.net = build_nn_from_config(input_dim=state_dim, output_dim=action_dim, nn_config=config["agents"][agent_name])
        self.max_action = max_action

    def forward(self, state):
        return torch.tanh(self.net(state)) * self.max_action


class Actor_TD3_discrete(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, agent_name, config):
        super().__init__()

        self.net = build_nn_from_config(input_dim=state_dim, output_dim=action_dim, nn_config=config["agents"][agent_name])
        self.max_action = max_action

    def forward(self, state):
        return torch.sigmoid(self.net(state)) * self.max_action


class Actor_PPO(nn.Module):
    def __init__(self, state_dim, action_dim, agent_name, config):
        super().__init__()

        self.net = build_nn_from_config(input_dim=state_dim, output_dim=action_dim, nn_config=config["agents"][agent_name])
        self.action_std = torch.nn.Parameter(torch.ones(action_dim, device=config["device"]) * config["agents"][agent_name]["action_std"])

    def forward(self, state):
        action_mean = torch.tanh(self.net(state))
        self.clamp(self.action_std, 0.001)
        dist = Normal(action_mean, self.action_std)
        return dist.rsample()

    def evaluate(self, state, action):
        action_mean = torch.tanh(self.net(state))
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

        self.net = build_nn_from_config(input_dim=state_dim + action_dim, output_dim=1, nn_config=config["agents"][agent_name])

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=len(state.shape) - 1))


class Critic_V(nn.Module):
    def __init__(self, state_dim, agent_name, config):
        super().__init__()

        self.net = build_nn_from_config(input_dim=state_dim, output_dim=1, nn_config=config["agents"][agent_name])

    def forward(self, state):
        return self.net(state)


class Critic_DQN(nn.Module):
    def __init__(self, state_dim, action_dim, agent_name, config):
        super().__init__()

        self.net = build_nn_from_config(input_dim=state_dim, output_dim=action_dim, nn_config=config["agents"][agent_name])

    def forward(self, state):
        return self.net(state)


class Critic_DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, agent_name, config):
        super().__init__()

        self.feature_stream = build_nn_from_config(input_dim=state_dim,
                                                   output_dim=config["agents"][agent_name]["feature_dim"],
                                                   nn_config=config["agents"][agent_name]
                                                   )

        heads_config = copy.copy(config["agents"][agent_name])
        heads_config["hidden_layer"] = 1
        heads_config["hidden_size"] = config["agents"][agent_name]["feature_dim"]

        self.value_stream = build_nn_from_config(input_dim=config["agents"][agent_name]["feature_dim"],
                                                 output_dim=1,
                                                 nn_config=heads_config
                                                 )

        self.advantage_stream = build_nn_from_config(input_dim=config["agents"][agent_name]["feature_dim"],
                                                     output_dim=action_dim,
                                                     nn_config=heads_config
                                                     )

    def forward(self, state):
        features = self.feature_stream(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean())
        return q_values
