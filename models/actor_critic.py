import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # gumbel_softmax_temp = config["agents"][agent_name]["gumbel_softmax_temp"]
        # self.gumbel_softmax_temp = torch.nn.Parameter(torch.tensor(gumbel_softmax_temp), requires_grad=True)
        self.gumbel_softmax_temp = config["agents"][agent_name]["gumbel_softmax_temp"]
        self.gumbel_softmax_hard = config["agents"][agent_name]["gumbel_softmax_hard"]

    def forward(self, state, tau):
        action = self.net(state) * self.max_action
        return F.gumbel_softmax(action, tau=tau, hard=self.gumbel_softmax_hard)


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


class Actor_SAC(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, agent_name, config):
        super().__init__()

        self.net = build_nn_from_config(input_dim=state_dim, output_dim=1, nn_config=config["agents"][agent_name])
        self.output_limit = max_action
        self.log_std_min = config["agents"][agent_name]['log_std_min']
        self.log_std_max = config["agents"][agent_name]['log_std_max']

        # Set output layers
        self.mu_layer = nn.Linear(action_dim, action_dim)
        self.log_std_layer = nn.Linear(action_dim, action_dim)

    def clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = (x > u).float()
        clip_low = (x < l).float()
        clip_value = (u - x) * clip_up + (l - x) * clip_low
        return x + clip_value.detach()

    def apply_squashing_func(self, mu, pi, log_pi):
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        log_pi -= torch.sum(torch.log(self.clip_but_pass_gradient(1 - pi.pow(2), l=0., u=1.) + 1e-6), dim=-1)
        return mu, pi, log_pi

    def forward(self, state):
        x = self.net(state)

        mu = self.mu_layer(x)
        log_std = torch.tanh(self.log_std_layer(x))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = torch.exp(log_std)

        # https://pytorch.org/docs/stable/distributions.html#normal
        dist = Normal(mu, std)
        pi = dist.rsample()  # Reparameterization trick (mean + std * N(0,1))
        log_pi = dist.log_prob(pi).sum(dim=-1)
        mu, pi, log_pi = self.apply_squashing_func(mu, pi, log_pi)

        # Make sure outputs are in correct range
        mu = mu * self.output_limit
        pi = pi * self.output_limit
        return mu, pi, log_pi
