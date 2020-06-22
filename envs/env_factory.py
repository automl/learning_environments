import torch
import gym
import numpy as np
from envs.virtual_env import VirtualEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EnvWrapper(gym.Wrapper):
    # wraps a gym/virtual environment for easier handling
    def __init__(self, env, is_virtual_env = False):
        super().__init__(env)
        self._env = env
        self.is_virtual_env = is_virtual_env
        self._max_episode_steps = env._max_episode_steps

    def step(self, action):
        if self.is_virtual_env:
            return super().step(action)
        else:
            next_state, reward, done, _ = super().step(action.detach().numpy())
            next_state_torch = torch.from_numpy(next_state).float().cpu()
            reward_torch = torch.tensor(reward, device="cpu", dtype=torch.float32)
            done_torch = torch.tensor(done, device="cpu", dtype=torch.float32)
            return next_state_torch, reward_torch, done_torch

    def reset(self):
        if self.is_virtual_env:
            return super().reset()
        else:
            return torch.from_numpy(super().reset()).float().cpu()

    def set_state(self, state):
        return self.env.set_state(state)

    def set_seed(self, seed):
        if self.is_virtual_env:
            self._env.set_seed(seed)
        else:
            print('Not a virtual environment, no need to set a seed')

    def random_action(self):
        # do random action in the [-1,1] range
        return torch.empty(self.get_action_dim(), device='cpu', dtype=torch.float32).uniform_(-1,1)

    def get_state_dim(self):
        if self.is_virtual_env:
            return self.state_dim
        else:
            return self.observation_space.shape[0]

    def get_action_dim(self):
        if self.is_virtual_env:
            return self.action_dim
        else:
            return self.action_space.shape[0]


class EnvFactory:
    def __init__(self, config):
        self.env_name = config['env_name']
        self.seed = config['seed']
        self.env_config = {}
        if self.env_name in config['envs'].keys():
            self.env_config = config['envs'][self.env_name]

        dummy_env = self.generate_default_real_env()
        self.state_dim = dummy_env.observation_space.shape[0]
        self.action_dim = dummy_env.action_space.shape[0]

    def generate_default_real_env(self):
        # generate a real environment with default parameters
        kwargs = self._get_default_parameters()
        print('Generating default real environment "{}" with parameters {}'.format(self.env_name, kwargs))
        env = self._env_factory(kwargs=kwargs)
        return EnvWrapper(env=env, is_virtual_env=False)

    def generate_random_real_env(self):
        # generate a real environment with random parameters within specified range
        kwargs = self._get_random_parameters()
        print('Generating random real environment "{}" with parameters {}'.format(self.env_name, kwargs))
        env = self._env_factory(kwargs=kwargs)
        return EnvWrapper(env=env, is_virtual_env=False)

    def generate_default_virtual_env(self):
        # generate a virtual environment with default parameters
        kwargs = self._get_default_parameters()
        print('Generating default virtual environment "{}" with parameters {}'.format(self.env_name, kwargs))
        env = VirtualEnv(state_dim=self.state_dim, action_dim=self.action_dim, env_name=self.env_name, **kwargs)
        return EnvWrapper(env=env, is_virtual_env=True)

    def _env_factory(self, kwargs):
        if self.env_name == "Pendulum-v0":
            env = gym.make(self.env_name)
            env._max_episode_steps = int(kwargs["max_steps"])
            env.max_speed = kwargs["max_speed"]
            env.max_torque = kwargs["max_torque"]
            env.g = kwargs["g"]
            env.m = kwargs["m"]
            env.l = kwargs["l"]
            env.dt = kwargs["dt"]
        else:
            raise NotImplementedError("Environment not supported")
        return env

    def _get_random_parameters(self):
        kwargs = {}
        for key, value in self.env_config.items():
            if isinstance(value, list):
                kwargs[key] = np.random.uniform(low=float(value[0]), high=float(value[2]))
            else:
                kwargs[key] = value
        return kwargs

    def _get_default_parameters(self):
        kwargs = {}
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
