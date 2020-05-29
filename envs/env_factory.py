import torch
import gym
import numpy as np
from envs.pendulum_env import PendulumEnv
from envs.virtual_env import VirtualEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EnvWrapper(gym.Wrapper):
    # wraps a gym/virtual environment for easier handling
    def __init__(self, env, is_virtual_env = False):
        super().__init__(env)
        self.is_virtual_env = is_virtual_env
        self._max_episode_steps = env._max_episode_steps

    def step(self, *args):
        # allows the step function to be either called with one (action) or two (action, state) parameters
        if len(args) == 1:      # action
            return self._step(args[0])
        elif len(args) == 2:    # action, state
            self.state = args[1]
            return self._step(args[0])
        else:
            raise NotImplementedError('Unknown number of input arguments')

    def _step(self, action):
        return super().step(action)

    def reset(self):
        return super().reset()

    def random_action(self):
        # do random action in the [-1,1] range
        # TODO: should be modified if environment has different range
        return torch.empty(self.get_action_dim(), device=device, dtype=torch.float32).uniform_(-1,1)

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
        self.virtual_config = config['envs']['virtual']
        self.env_config = {}
        if self.env_name in config['envs'].keys():
            self.env_config = config['envs'][self.env_name]

    def generate_default_env(self):
        # generate a real environment with default parameters
        kwargs = {}
        for key, value in self.env_config.items():
            kwargs[key] = float(value[1])

        return self.env_factory(kwargs = kwargs)

    def generate_real_env(self):
        # generate a real environment with random parameters within specified range
        kwargs = {}
        for key, value in self.env_config.items():
            kwargs[key] = np.random.uniform(low=float(value[0]), high=float(value[2]))

        return self.env_factory(kwargs)

    def generate_virtual_env(self):
        dummy_env = self.generate_default_env()
        state_dim = dummy_env.observation_space.shape[0]
        action_dim = dummy_env.action_space.shape[0]

        kwargs = {}
        for key, value in self.env_config.items():
            kwargs[key] = float(value[1])

        env = VirtualEnv(state_dim = state_dim,
                         action_dim = action_dim,
                         kwargs = kwargs)

        return EnvWrapper(env = env,
                          is_virtual_env = True)

    def env_factory(self, kwargs):
        if self.env_name == 'PendulumEnv':
            print('Generating environment "{}" with parameters {}'.format(self.env_name, kwargs))
            env = PendulumEnv(**kwargs)
        else:
            raise NotImplementedError('Environment not supported')
        env.seed(self.seed)

        return EnvWrapper(env = env,
                          is_virtual_env = False)


if __name__ == "__main__":
    import yaml

    with open("../default_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
        gen = EnvFactory(config)
        a = gen.generate_real_env()
