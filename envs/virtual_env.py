import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VirtualEnv(nn.Module):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(VirtualEnv, self).__init__()
        self.state_dim = state_dim
        self.seed = torch.Tensor(np.random.random(), device=device)
        self.state = torch.zeros(state_dim, device=device)

        hidden_size = kwargs['hidden_size']
        self.net = nn.Sequential(
                        nn.Linear(state_dim+action_dim+1, hidden_size), # +1 because of seed
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, state_dim+2))            # +2 because of reward/done

    def set_seed(self, seed):
        self.seed = torch.Tensor(seed, device=device)

    def reset(self):
        self.state = torch.zeros(self.state_dim, device=device)
        return self.state

    def step(self, action):
        input = torch.cat((torch.Tensor(action), self.state, self.seed))
        output = self.net(input)
        self.state = output[:self.state_dim]
        output = output.cpu().data.numpy()
        next_state = output[:self.state_dim]
        reward = output[-2]
        done = output[-1]
        return next_state, reward, done


