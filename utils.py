import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_nn_from_config(input_dim, output_dim, config):
    hidden_size = config['ac_model']['hidden_size']
    activation_fn = config['ac_model']['activation_fn']

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
    def __init__(self, state_dim, action_dim, config):
        super(Actor, self).__init__()

        self.net = build_nn_from_config(input_dim = state_dim,
                                        output_dim = action_dim,
                                        config = config)
        self.action_std = torch.nn.Parameter(
            torch.ones(action_dim, device=device)*config['ac_model']['action_std'])

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
    def __init__(self, state_dim, action_dim, config):
        super(Critic_Q, self).__init__()

        self.net = build_nn_from_config(input_dim = state_dim+action_dim,
                                        output_dim = action_dim,
                                        config = config)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))


class Critic_V(nn.Module):
    def __init__(self, state_dim, config):
        super(Critic_V, self).__init__()

        self.net = build_nn_from_config(input_dim = state_dim,
                                        output_dim = 1,
                                        config = config)

    def forward(self, state):
        return torch.squeeze(self.net(state))


class ReplayBuffer:
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


class AverageMeter:
    def __init__(self, buffer_size, update_rate, print_str):
        self.buffer_size = buffer_size
        self.update_rate = update_rate
        self.print_str = print_str
        self.vals = np.zeros(buffer_size)
        self.ptr = 0
        self.size = 0
        self.it = 1

    def update(self, val):
        self.vals[self.ptr] = val
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        self.it += 1

        if self.it % self.update_rate == 0:
            print(self.print_str + str(np.mean(self.vals[:self.size])) + '   Total updates: ' + str(self.it))
