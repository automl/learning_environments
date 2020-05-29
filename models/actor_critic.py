import torch
import torch.nn as nn
from torch.distributions.normal import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_nn_from_config(input_dim, output_dim, agent_name, config):
    hidden_size = config['agents'][agent_name]['hidden_size']
    activation_fn = config['agents'][agent_name]['activation_fn']

    if activation_fn == 'relu':
        act_fn = nn.ReLU()
    elif activation_fn == 'tanh':
        act_fn = nn.Tanh()

    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        act_fn,
        nn.Linear(hidden_size, hidden_size),
        act_fn,
        nn.Linear(hidden_size, output_dim))


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, agent_name, config):
        super(Actor, self).__init__()

        self.net = build_nn_from_config(input_dim = state_dim,
                                        output_dim = action_dim,
                                        agent_name = agent_name,
                                        config = config)
        self.action_std = torch.nn.Parameter(
            torch.ones(action_dim, device=device)*config['agents'][agent_name]['action_std'])

    def forward(self, state):
        action_mean = self.net(state)
        dist = Normal(action_mean, self.action_std)
        return dist.rsample()

    def evaluate(self, state, action):
        action_mean = self.net(state)
        dist = Normal(action_mean, self.action_std)
        action_logprobs = torch.sum(dist.log_prob(action), dim=1)
        dist_entropy = torch.sum(dist.entropy(), dim=1)
        return action_logprobs, dist_entropy


class Critic_Q(nn.Module):
    def __init__(self, state_dim, action_dim, agent_name, config):
        super(Critic_Q, self).__init__()

        self.net = build_nn_from_config(input_dim = state_dim+action_dim,
                                        output_dim = action_dim,
                                        agent_name=agent_name,
                                        config = config)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))


class Critic_V(nn.Module):
    def __init__(self, state_dim, agent_name, config):
        super(Critic_V, self).__init__()

        self.net = build_nn_from_config(input_dim = state_dim,
                                        output_dim = 1,
                                        agent_name=agent_name,
                                        config = config)

    def forward(self, state):
        return self.net(state)
