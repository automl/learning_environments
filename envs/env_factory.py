import torch
from envs.bandit import *
from envs.gridworld import *
from envs.virtual_env import VirtualEnv
from envs.reward_env import RewardEnv
from envs.env_wrapper import EnvWrapper
from gym.wrappers import TimeLimit


class EnvFactory:
    def __init__(self, config):
        self.env_name = config["env_name"]
        self.device = config["device"]
        self.env_config = config["envs"][self.env_name]

        # for virtual env
        dummy_env = self.generate_real_env(print_str='EnvFactory (dummy_env): ')
        self.state_dim = dummy_env.get_state_dim()
        self.action_dim = dummy_env.get_action_dim()
        self.observation_space = dummy_env.env.observation_space
        self.action_space = dummy_env.env.action_space

    def generate_real_env(self, print_str=''):
        # generate a real environment with default parameters
        kwargs = self._get_default_parameters(virtual_env=False)
        #print(print_str + 'Generating default real environment "{}" with parameters {}'.format(self.env_name, kwargs))
        env = self._generate_real_env_with_kwargs(kwargs=kwargs, env_name=self.env_name)
        return EnvWrapper(env=env)

    def generate_virtual_env(self, print_str=''):
        # generate a virtual environment with default parameters
        kwargs = self._get_default_parameters(virtual_env=True)
        #print(print_str + 'Generating virtual environment "{}" with parameters {}'.format(self.env_name, kwargs))
        env = VirtualEnv(kwargs)
        return EnvWrapper(env=env).to(self.device)

    def generate_reward_env(self, print_str=''):
        # generate a virtual environment with default parameters
        kwargs = self._get_default_parameters(virtual_env=True)
        #print(print_str + 'Generating reward environment "{}" with parameters {}'.format(self.env_name, kwargs))
        real_env = self._generate_real_env_with_kwargs(kwargs=kwargs, env_name=self.env_name)
        reward_env = RewardEnv(real_env=real_env, kwargs=kwargs)
        return EnvWrapper(env=reward_env).to(self.device)

    def _get_default_parameters(self, virtual_env):
        kwargs = {"env_name": self.env_name,
                  "device": self.device}
        if virtual_env:
            kwargs["state_dim"] = self.state_dim
            kwargs["action_dim"] = self.action_dim
            kwargs["observation_space"] = self.observation_space
            kwargs["action_space"] = self.action_space
            kwargs["reset_env"] = self.generate_real_env()
        for key, value in self.env_config.items():
            if isinstance(value, list):
                kwargs[key] = float(value[1])
            else:
                kwargs[key] = value
        return kwargs

    def _generate_real_env_with_kwargs(self, kwargs, env_name):
        # generate environment class
        # todo: make generic (e.g. check if class is existent in bandit.py or gridworld.py)
        if env_name == "Bandit":
            env = TimeLimit(BanditFixedPermutedGaussian())
        elif env_name == "EmptyRoom22":
            env = TimeLimit(EmptyRoom22())
        elif env_name == "EmptyRoom23":
            env = TimeLimit(EmptyRoom23())
        elif env_name == "EmptyRoom33":
            env = TimeLimit(EmptyRoom33())
        elif env_name == "WallRoom":
            env = TimeLimit(WallRoom())
        elif env_name == "HoleRoom":
            env = TimeLimit(HoleRoom())
        elif env_name == "HoleRoomLarge":
            env = TimeLimit(HoleRoomLarge())
        elif env_name == "HoleRoomLargeShifted":
            env = TimeLimit(HoleRoomLargeShifted())
        elif env_name == "Cliff":
            env = TimeLimit(Cliff())
        else:
            env = gym.make(env_name)

        for key, value in kwargs.items():
            setattr(env, key, value)

        # for episode termination
        env._max_episode_steps = int(kwargs["max_steps"])
        # for model save/load
        env.kwargs = kwargs

        return env


if __name__ == "__main__":
    import yaml

    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        gen = EnvFactory(config)
        a = gen.generate_real_env()
