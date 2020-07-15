import os
import gym
import torch
import torch.nn as nn
import numpy as np
from models.model_utils import build_nn_from_config
from envs.env_utils import generate_env_with_kwargs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VirtualEnv(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.input_seed = None
        self.env_name = str(kwargs["env_name"])
        self.state_dim = int(kwargs["state_dim"])
        self.action_dim = int(kwargs["action_dim"])
        self.zero_init = bool(kwargs["zero_init"])
        self.solved_reward = float(kwargs["solved_reward"])
        self.state = torch.zeros(self.state_dim, device=device)  # not sure what the default state should be

        # for compatibility
        self._max_episode_steps = int(kwargs["max_steps"])

        # for rendering
        self.viewer_env = generate_env_with_kwargs(kwargs, self.env_name)
        self.viewer_env.reset()

        self.state_net = build_nn_from_config(input_dim = self.state_dim + self.action_dim + 1,
                                              output_dim = self.state_dim,
                                              nn_config = kwargs).to(device)
        self.reward_net = build_nn_from_config(input_dim = self.state_dim + self.action_dim + 1,
                                               output_dim = 1,
                                               nn_config = kwargs).to(device)
        self.done_net = build_nn_from_config(input_dim = self.state_dim + self.action_dim + 1,
                                             output_dim = 1,
                                             nn_config = kwargs).to(device)

    def reset(self):
        if self.zero_init:
            state = torch.zeros(self.state_dim, device=device)
        elif self.env_name == "Pendulum-v0" or self.env_name == "Test":
            state = torch.tensor(
                [np.random.uniform(low=-1, high=1), np.random.uniform(low=-1, high=1), np.random.uniform(low=-1, high=1)],
                device=device,
                dtype=torch.float32,
            )
        else:
            raise NotImplementedError("Unknown environment: non-zero state reset only for supported environments.")

        return state

    def step(self, action, state, input_seed):
        input = torch.cat((action, state, input_seed), dim=len(action.shape) - 1)
        next_state = self.state_net(input)
        reward = self.reward_net(input)
        done = (self.done_net(input) > 0.5).float()
        return next_state, reward, done

    def render(self, state):
        if self.env_name != 'Test':
            self.viewer_env.state = state.cpu().detach().numpy()
            self.viewer_env.render()

    def close(self):
        if self.env_name != 'Test':
            if self.viewer_env.viewer:
                self.viewer_env.viewer.close()
                self.viewer_env.viewer = None

    def get_state_dict(self):
        env_state = {}
        env_state['virtual_env_state_net'] = self.state_net.state_dict()
        env_state['virtual_env_reward_net'] = self.reward_net.state_dict()
        env_state['virtual_env_done_net'] = self.done_net.state_dict()
        return env_state

    def set_state_dict(self, env_state):
        self.state_net.load_state_dict(env_state['virtual_env_state_net'])
        self.reward_net.load_state_dict(env_state['virtual_env_reward_net'])
        self.done_net.load_state_dict(env_state['virtual_env_done_net'])




