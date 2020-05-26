import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions.normal import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_nn_config(config):
    hidden_size = config['model']['hidden_size']
    action_std = config['model']['action_std']
    activation_fn = config['model']['activation_fn']

    if activation_fn == 'relu':
        return hidden_size, action_std, nn.ReLU()
    elif activation_fn == 'tanh':
        return hidden_size, action_std, nn.Tanh()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(Actor, self).__init__()

        hidden_size, action_std, activation_fn = get_nn_config(config)

        self.net = nn.Sequential(
                        nn.Linear(state_dim, hidden_size),
                        activation_fn,
                        nn.Linear(hidden_size, hidden_size),
                        activation_fn,
                        nn.Linear(hidden_size, action_dim),
                        nn.Tanh())

        self.action_std = torch.nn.Parameter(torch.ones(action_dim)*action_std).to(device)

    def forward(self, state):
        action_mean = self.net(state)
        dist = Normal(action_mean, self.action_std)
        action = dist.rsample()
        return action

    def evaluate(self, state, action):
        action_mean = self.net(state)
        dist = Normal(action_mean, self.action_std)
        action_logprobs = torch.sum(dist.log_prob(action), dim=1)
        dist_entropy = torch.sum(dist.entropy(), dim=1)
        return action_logprobs, dist_entropy


class Critic_Q(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(Critic_Q, self).__init__()

        hidden_size, action_std, activation_fn = get_nn_config(config)

        self.net = nn.Sequential(
                        nn.Linear(state_dim + action_dim, hidden_size),
                        activation_fn,
                        nn.Linear(hidden_size, hidden_size),
                        activation_fn,
                        nn.Linear(hidden_size, action_dim))

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))


class Critic_V(nn.Module):
    def __init__(self, state_dim, config):
        super(Critic_V, self).__init__()

        hidden_size, action_std, activation_fn = get_nn_config(config)

        self.net = nn.Sequential(
                        nn.Linear(state_dim, hidden_size),
                        activation_fn,
                        nn.Linear(hidden_size, hidden_size),
                        activation_fn,
                        nn.Linear(hidden_size, 1))

    def forward(self, state):
        return torch.squeeze(self.net(state))


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))
        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # for TD3
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )

    # for PPO
    def get_all(self):
        return (
            torch.FloatTensor(self.state[:self.size]).to(self.device),
            torch.FloatTensor(self.action[:self.size]).to(self.device),
            torch.FloatTensor(self.next_state[:self.size]).to(self.device),
            torch.FloatTensor(self.reward[:self.size]).to(self.device),
            torch.FloatTensor(self.done[:self.size]).to(self.device)
        )

    # for PPO
    def clear(self):
        self.__init__(self.state_dim, self.action_dim, self.max_size)
