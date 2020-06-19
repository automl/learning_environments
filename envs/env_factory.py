import torch
import gym
import numpy as np
from envs.pendulum_env import PendulumEnv
from envs.virtual_env import VirtualEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EnvWrapper(gym.Wrapper):
    # wraps a gym/virtual environment for easier handling
    def __init__(self, env, is_virtual_env=False):
        super().__init__(env)
        self.is_virtual_env = is_virtual_env
        self._max_episode_steps = env._max_episode_steps

    def step(self, *args):
        # allows the step function to be either called with one (action) or two (action, state) parameters
        if len(args) == 1:  # action
            return self._step(args[0])
        elif len(args) == 2:  # action, state
            self.state = args[1]
            return self._step(args[0])
        else:
            raise NotImplementedError("Unknown number of input arguments")

    def _step(self, action):
        return super().step(action)

    def reset(self):
        return super().reset()

    def set_state(self, state):
        return self.env.set_state(state)

    def random_action(self):
        # do random action in the [-2,2] range
        # todo: should be modified if environment has different range
        return torch.empty(self.get_action_dim(), device="cpu", dtype=torch.float32).uniform_(-2, 2)

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
        self.env_name = config["env_name"]
        self.seed = config["seed"]
        self.env_config = {}
        if self.env_name in config["envs"].keys():
            self.env_config = config["envs"][self.env_name]

        dummy_env = self.generate_default_real_env()
        self.state_dim = dummy_env.observation_space.shape[0]
        self.action_dim = dummy_env.action_space.shape[0]

    def generate_default_real_env(self):
        # generate a real environment with default parameters
        kwargs = self._get_default_parameters()
        print('Generating default real environment "{}" with parameters {}'.format(self.env_name, kwargs))
        return self._env_factory(kwargs=kwargs)

    def generate_random_real_env(self):
        # generate a real environment with random parameters within specified range
        kwargs = self._get_random_parameters()
        print('Generating random real environment "{}" with parameters {}'.format(self.env_name, kwargs))
        return self._env_factory(kwargs=kwargs)

    def generate_default_virtual_env(self):
        # generate a virtual environment with default parameters
        kwargs = self._get_default_parameters()
        print('Generating default virtual environment "{}" with parameters {}'.format(self.env_name, kwargs))
        env = VirtualEnv(state_dim=self.state_dim, action_dim=self.action_dim, **kwargs)
        return EnvWrapper(env=env, is_virtual_env=True)

    def _env_factory(self, kwargs):
        if self.env_name == "PendulumEnv":
            env = PendulumEnv(**kwargs)
        else:
            raise NotImplementedError("Environment not supported")
        env.seed(self.seed)

        return EnvWrapper(env=env, is_virtual_env=False)

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

        kwargs["env_name"] = self.env_name
        return kwargs


if __name__ == "__main__":
    import yaml

    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        gen = EnvFactory(config)
        a = gen.generate_default_real_env()
