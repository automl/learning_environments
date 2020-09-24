import torch
import torch.nn as nn


class RBF(nn.Module):
    def __init__(self, nodes):
        super(RBF, self).__init__()
        self.eps = nn.Parameter(torch.ones(nodes))
        self.mean = nn.Parameter(torch.zeros(nodes))
        self.counter = 0

    def forward(self, x):
        self.counter += 1

        if self.counter % 5000 == 0:
            print(sum(self.eps))

        return torch.exp(-(x*self.eps).pow(2))


class Identity(nn.Module):
    def __init__(self, module):
        super(Identity, self).__init__()
        self.net = module

    def forward(self, x):
        return self.net(x)


def build_nn_from_config(input_dim, output_dim, nn_config):
    hidden_size = nn_config['hidden_size']
    hidden_layer = nn_config['hidden_layer']
    activation_fn = nn_config['activation_fn']

    if activation_fn == 'prelu':
        act_fn = nn.PReLU()
    elif activation_fn == 'relu':
        act_fn = nn.ReLU()
    elif activation_fn == 'leakyrelu':
        act_fn = nn.LeakyReLU()
    elif activation_fn == 'tanh':
        act_fn = nn.Tanh()
    else:
        print('Unknown activation function')

    modules = []
    modules.append(nn.Linear(input_dim, hidden_size))
    modules.append(act_fn)
    for i in range(hidden_layer-1):
        modules.append(nn.Linear(hidden_size, hidden_size))
        modules.append(act_fn)
    modules.append(nn.Linear(hidden_size, output_dim))
    return nn.Sequential(*modules)
