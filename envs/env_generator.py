import torch
import torch.nn as nn
import gym
import numpy as np
from envs.pendulum import PendulumEnv
from envs.virtual import VirtualEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EnvWrapper:
    def __init__


class EnvGenerator:
    def __init__(self, config):
        self.env_name = config['env_name']

        self.env_config = {}
        if self.env_name in config['envs'].keys():
            self.env_config = config['envs'][self.env_name]

    def generate_default_env(self):
        # generate a real environment with default parameters
        kwargs = {}
        for key, value in self.env_config.items():
            kwargs[key] = float(value[1])
        return self.env_factory(kwargs)

    def generate_real_env(self):
        # generate a real environment with random parameters within specified range
        kwargs = {}
        for key, value in self.env_config.items():
            kwargs[key] = np.random.uniform(low=float(value[0]), high=float(value[2]))
        return self.env_factory(kwargs)

    def generate_virtual_env(self):
        # TODO: generate a more useful neural net
        dummy_env = self.generate_default_env()
        state_dim = dummy_env.observation_space.shape[0]
        action_dim = dummy_env.action_space.shape[0]
        return VirtualEnv(state_dim = state_dim,
                          action_dim = action_dim,
                          seed = np.random.random())

    def env_factory(self, kwargs):
        if self.env_name == 'PendulumEnv':
            return PendulumEnv(**kwargs)
        else:
            return gym.make(self.env_name)


if __name__ == "__main__":
    import yaml

    with open("../default_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
        gen = EnvGenerator(config)
        a = gen.generate_real_env()
