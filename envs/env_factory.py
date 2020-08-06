import numpy as np
import gym
import torch
from envs.env_wrapper import EnvWrapper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EnvFactory:
    def __init__(self, config):
        self.env_name = config["env_name"]
        self.env_config = config["envs"][self.env_name]

        dummy_env = self.generate_default_real_env()
        self.state_dim = dummy_env.get_state_dim()
        self.action_dim = dummy_env.get_action_dim()

    def generate_default_real_env(self):
        # generate a real environment with default parameters
        kwargs = self._get_default_parameters()
        print('Generating default real environment "{}" with parameters {}'.format(self.env_name, kwargs))
        env = self._generate_env_with_kwargs(kwargs=kwargs, env_name=self.env_name)
        return EnvWrapper(env=env)

    def generate_random_real_env(self):
        # generate a real environment with random parameters within specified range
        kwargs = self._get_random_parameters()
        print('Generating random real environment "{}" with parameters {}'.format(self.env_name, kwargs))
        env = self._generate_env_with_kwargs(kwargs=kwargs, env_name=self.env_name)
        return EnvWrapper(env=env)

    def generate_interpolated_real_env(self, interpolate):
        # generate a real environment with random parameters within specified range
        kwargs = self._get_interpolate_parameters(interpolate)
        print('Generating interpolated real environment "{}" with parameters {}'.format(self.env_name, kwargs))
        env = self._generate_env_with_kwargs(kwargs=kwargs, env_name=self.env_name)
        return EnvWrapper(env=env)

    def _get_random_parameters(self):
        kwargs = {"env_name": self.env_name, "state_dim": self.state_dim, "action_dim": self.action_dim}
        for key, value in self.env_config.items():
            if isinstance(value, list):
                kwargs[key] = np.random.uniform(low=float(value[0]), high=float(value[2]))
            else:
                kwargs[key] = value
        return kwargs

    def _get_interpolate_parameters(self, interpolate):
        kwargs = {"env_name": self.env_name, "state_dim": self.state_dim, "action_dim": self.action_dim}
        for key, value in self.env_config.items():
            if isinstance(value, list):
                if bool(value[3]):
                    kwargs[key] = value[0] + interpolate * (value[2] - value[0])
                else:
                    kwargs[key] = value[2] + interpolate * (value[0] - value[2])
            else:
                kwargs[key] = value
        return kwargs

    def _get_default_parameters(self):
        kwargs = {"env_name": self.env_name}
        for key, value in self.env_config.items():
            if isinstance(value, list):
                kwargs[key] = float(value[1])
            else:
                kwargs[key] = value
        return kwargs

    def _generate_env_with_kwargs(self, kwargs, env_name):
        # generate environment class
        if env_name == "Pendulum-v0":
            env = gym.make("Pendulum-v0")
        elif env_name == "MountainCarContinuous-v0":
            env = gym.make("MountainCarContinuous-v0")
        elif env_name == "HalfCheetah-v2":
            env = gym.make("HalfCheetah-v2")
        else:
            raise NotImplementedError("Environment not supported")

        # set environment parameters
        if env_name == "HalfCheetah-v2":
            for key, value in kwargs.items():
                if "g" == key:  # gravity along negative z-axis
                    env.model.opt.gravity[2] = value
                elif "cripple_joint" == key:
                    if value:  # cripple_joint True
                        env.cripple_mask = np.ones(env.action_space.shape)
                        idx = np.random.choice(env.action_space.shape[0])
                        env.cripple_mask[idx] = 0
                else:
                    setattr(env, key, value)
        else:
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
        a = gen.generate_default_real_env()
