import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VirtualEnv(nn.Module):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(VirtualEnv, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seed = torch.FloatTensor(np.random.random(), device=device)
        self.state = torch.zeros(state_dim, device=device)

        hidden_size = kwargs['hidden_size']

        self.base = nn.Sequential(
                        nn.Linear(state_dim+action_dim+1, hidden_size), # +1 because of seed
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU())                                      # +2 because of reward/done
        self.state_head = nn.Linear(hidden_size, state_dim)
        self.reward_head = nn.Linear(hidden_size, 1)
        self.done_head = nn.Linear(hidden_size, 1)

    def set_seed(self, seed):
        self.seed = torch.FloatTensor(seed, device=device)

    def reset(self):
        self.state = torch.zeros(self.state_dim, device=device)
        return self.state

    def step(self, action):
        input = torch.cat((action, self.state, self.seed))
        x = self.base(input)
        next_state = self.state_head(x)
        reward = self.reward_head(x)
        done = self.done_head(x)
        return next_state, reward, done