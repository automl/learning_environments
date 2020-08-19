import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Identity(nn.Module):
    def __init__(self, module):
        super(Identity, self).__init__()
        self.net = module

    def forward(self, x):
        return self.net(x)


class Dropout(nn.Module):
    def __init__(self, input_dim, p):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.input_dim = input_dim
        self.reset_dropout()

    def reset_dropout(self):
        self.dropout = torch.distributions.binomial.Binomial(probs=1-self.p).sample((self.input_dim,)).to(device)

    def get_dropout(self):
        return self.dropout

    def set_dropout(self, dropout):
        self.dropout = dropout

    def forward(self, X):
        if self.training:
            return X * self.dropout * (1.0/(1-self.p))
        else:
            return X


def build_nn_from_config(input_dim, output_dim, nn_config):
    hidden_size = nn_config['hidden_size']
    hidden_layer = nn_config['hidden_layer']
    activation_fn = nn_config['activation_fn']
    weight_norm = nn_config['weight_norm']
    dropout = nn_config['dropout']

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
    modules.append(Dropout(hidden_size, dropout))
    modules.append(act_fn)
    for i in range(hidden_layer):
        modules.append(weight_nrm(nn.Linear(hidden_size, hidden_size)))
        modules.append(Dropout(hidden_size, dropout))
        modules.append(act_fn)
    modules.append(weight_nrm(nn.Linear(hidden_size, output_dim)))
    return nn.Sequential(*modules)
