import gym
import numpy as np
import torch
import torch.nn as nn
from envs.virtual_env import VirtualEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EnvWrapper(nn.Module):
    # wraps a gym/virtual environment for easier handling
    def __init__(self, env):
        super().__init__()
        self.env = env

    def step(self, action):
        if isinstance(self.env, VirtualEnv):
            return self.env.step(action)
        else:
            next_state, reward, done, _ = self.env.step(action.detach().numpy())
            next_state_torch = torch.from_numpy(next_state).float().cpu()
            reward_torch = torch.tensor(reward, device="cpu", dtype=torch.float32)
            done_torch = torch.tensor(done, device="cpu", dtype=torch.float32)
            return next_state_torch, reward_torch, done_torch

    def reset(self):
        if isinstance(self.env, VirtualEnv):
            return self.env.reset()
        else:
            return torch.from_numpy(self.env.reset()).float().cpu()

    def set_seed(self, seed):
        if isinstance(self.env, VirtualEnv):
            return self.env.set_seed(seed)
        else:
            print('Not a virtual environment, no need to set a seed')

    def get_random_action(self):
        # do random action in the [-1,1] range
        return torch.empty(self.get_action_dim(), device='cpu', dtype=torch.float32).uniform_(-1,1)

    def get_state_dim(self):
        if isinstance(self.env, VirtualEnv):
            return self.env.state_dim
        else:
            return self.env.observation_space.shape[0]

    def get_action_dim(self):
        if isinstance(self.env, VirtualEnv):
            return self.env.action_dim
        else:
            return self.env.action_space.shape[0]

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def max_episode_steps(self):
        return self.env._max_episode_steps

    def seed(self, seed):
        if not isinstance(self.env, VirtualEnv):
            return self.env.seed()


class EnvFactory:
    def __init__(self, config):
        self.env_name = config['env_name']
        self.env_config = {}
        if self.env_name in config['envs'].keys():
            self.env_config = config['envs'][self.env_name]

        dummy_env = self.generate_default_real_env()
        self.state_dim = dummy_env.get_state_dim()
        self.action_dim = dummy_env.get_action_dim()

    def generate_default_real_env(self):
        # generate a real environment with default parameters
        kwargs = self._get_default_parameters()
        print('Generating default real environment "{}" with parameters {}'.format(self.env_name, kwargs))
        env = self._env_factory(kwargs=kwargs)
        return EnvWrapper(env=env)

    def generate_random_real_env(self):
        # generate a real environment with random parameters within specified range
        kwargs = self._get_random_parameters()
        print('Generating random real environment "{}" with parameters {}'.format(self.env_name, kwargs))
        env = self._env_factory(kwargs=kwargs)
        return EnvWrapper(env=env)

    def generate_default_virtual_env(self):
        # generate a virtual environment with default parameters
        kwargs = self._get_default_parameters()
        print('Generating default virtual environment "{}" with parameters {}'.format(self.env_name, kwargs))
        env = VirtualEnv(**kwargs)
        return EnvWrapper(env=env)

    def _env_factory(self, kwargs):
        if self.env_name == "Pendulum-v0" or self.env_name == 'MountainCarContinuous-v0':
            env = gym.make(self.env_name)
        else:
            raise NotImplementedError("Environment not supported")

        for key, value in kwargs.items():
            setattr(env, key, value)

        env._max_episode_steps = int(kwargs["max_steps"])

        return env

    def _get_random_parameters(self):
        kwargs = {}
        kwargs['env_name'] = self.env_name
        for key, value in self.env_config.items():
            if isinstance(value, list):
                kwargs[key] = np.random.uniform(low=float(value[0]), high=float(value[2]))
            else:
                kwargs[key] = value
        return kwargs

    def _get_default_parameters(self):
        kwargs = {}
        kwargs['env_name'] = self.env_name
        for key, value in self.env_config.items():
            if isinstance(value, list):
                kwargs[key] = float(value[1])
            else:
                kwargs[key] = value
        return kwargs


if __name__ == "__main__":
    import yaml

    with open("../default_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
        gen = EnvFactory(config)
        a = gen.generate_default_real_env()
