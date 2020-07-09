import torch
import torch.nn as nn
from models.model_utils import build_nn_from_config
from torch.distributions.normal import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, agent_name, config):
        super().__init__()

        self.net = build_nn_from_config(input_dim=state_dim, output_dim=action_dim, nn_config=config["agents"][agent_name])

        self.action_std = torch.nn.Parameter(torch.ones(action_dim, device=device) * config["agents"][agent_name]["action_std"])

    def forward(self, state):
        action_mean = self.net(state)
        self._check_and_correct_valid_std()
        dist = Normal(action_mean, self.action_std)
        return dist.rsample()

    def reconstruct_autograd(self, state, std):
        rng = [torch.manual_seed(abs(int(sum(state[i]) * 1e9))) for i in range(state.shape[0])]
        means = self.net(state)
        self._check_and_correct_valid_std()
        dist = Normal(means, std, rng)
        return dist.rsample()

    def evaluate(self, state, action):
        action_mean = self.net(state)
        dist = Normal(action_mean, self.action_std)
        action_logprobs = torch.sum(dist.log_prob(action), dim=1)
        dist_entropy = torch.sum(dist.entropy(), dim=1)
        return action_logprobs, dist_entropy

    def _check_and_correct_valid_std(self):
        for i in range(self.action_std.shape[0]):
            if self.action_std[i] < 0:
                print("action std < 0 -> clamped.")
        self.action_std = torch.nn.Parameter(torch.clamp(self.action_std, 0.0))


class Critic_Q(nn.Module):
    def __init__(self, state_dim, action_dim, agent_name, config):
        super().__init__()

        self.net = build_nn_from_config(input_dim=state_dim + action_dim, output_dim=action_dim, nn_config=config["agents"][agent_name])

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))


class Critic_V(nn.Module):
    def __init__(self, state_dim, agent_name, config):
        super().__init__()

        self.net = build_nn_from_config(input_dim=state_dim, output_dim=1, nn_config=config["agents"][agent_name])

    def forward(self, state):
        return self.net(state)
