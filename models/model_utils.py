import torch
import torch.nn as nn

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
    weight_norm = nn_config['weight_norm']

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

    if weight_norm:
        weight_nrm = torch.nn.utils.weight_norm
    else:
        weight_nrm = Identity

    modules = []
    modules.append(weight_nrm(nn.Linear(input_dim, hidden_size)))
    modules.append(act_fn)
    for i in range(hidden_layer):
        modules.append(weight_nrm(nn.Linear(hidden_size, hidden_size)))
        modules.append(act_fn)
    modules.append(weight_nrm(nn.Linear(hidden_size, output_dim)))
    return nn.Sequential(*modules)
