import torch
import torch.nn as nn
from models.model_utils import build_nn_from_config


class VirtualEnv(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.input_seed = None
        self.env_name = str(kwargs["env_name"])
        self.device = str(kwargs["device"])
        self.state_dim = int(kwargs["state_dim"])
        self.action_dim = int(kwargs["action_dim"])
        self.solved_reward = int(kwargs["solved_reward"])

        self.state = self.reset()

        # for gym compatibility
        self._max_episode_steps = int(kwargs["max_steps"])
        self.action_space = kwargs["action_space"]
        self.observation_space = kwargs["observation_space"]

        self.state_net = build_nn_from_config(input_dim=self.state_dim + self.action_dim,
                                              output_dim=self.state_dim,
                                              nn_config=kwargs).to(self.device)
        self.reward_net = build_nn_from_config(input_dim=self.state_dim + self.action_dim,
                                               output_dim=1,
                                               nn_config=kwargs).to(self.device)
        self.done_net = build_nn_from_config(input_dim=self.state_dim + self.action_dim,
                                             output_dim=1,
                                             nn_config=kwargs).to(self.device)

    def reset(self):
        dist = torch.distributions.Normal(torch.zeros(self.state_dim), torch.ones(self.state_dim))
        self.state = dist.sample()
        self.state = torch.zeros(self.state_dim)
        return self.state

    def step(self, action):
        if self.action_dim > 1 and len(action) == 1:
            action = self.to_one_hot_encoding(action)
        input = torch.cat((action.to(self.device), self.state.to(self.device)), dim=len(action.shape) - 1)
        next_state = self.state_net(input)
        reward = self.reward_net(input)
        done = self.done_net(input)
        return next_state, reward, done

    def to_one_hot_encoding(self, action):
        action_one_hot = torch.zeros(self.action_dim)
        action_one_hot[int(action.item())] = 1
        return action_one_hot

    def get_state_dict(self):
        env_state = {}
        env_state["virtual_env_state_net"] = self.state_net.state_dict()
        env_state["virtual_env_reward_net"] = self.reward_net.state_dict()
        env_state["virtual_env_done_net"] = self.done_net.state_dict()
        return env_state

    def set_state_dict(self, env_state):
        self.state_net.load_state_dict(env_state["virtual_env_state_net"])
        self.reward_net.load_state_dict(env_state["virtual_env_reward_net"])
        self.done_net.load_state_dict(env_state["virtual_env_done_net"])
