import numpy as np
import torch
from envs.virtual_env import VirtualEnv
from envs.env_wrapper import EnvWrapper
from envs.env_utils import generate_env_with_kwargs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EnvFactory:
    def __init__(self, config):
        self.env_name = config["env_name"]
        self.env_config = {}
        if self.env_name in config["envs"].keys():
            self.env_config = config["envs"][self.env_name]

        dummy_env = self.generate_default_real_env()
        self.state_dim = dummy_env.get_state_dim()
        self.action_dim = dummy_env.get_action_dim()

    def generate_default_real_env(self):
        # generate a real environment with default parameters
        kwargs = self._get_default_parameters(virtual_env = False)
        print('Generating default real environment "{}" with parameters {}'.format(self.env_name, kwargs))
        env = generate_env_with_kwargs(kwargs=kwargs, env_name=self.env_name)
        return EnvWrapper(env=env)

    def generate_random_real_env(self):
        # generate a real environment with random parameters within specified range
        kwargs = self._get_random_parameters()
        print('Generating random real environment "{}" with parameters {}'.format(self.env_name, kwargs))
        env = generate_env_with_kwargs(kwargs=kwargs, env_name=self.env_name)
        return EnvWrapper(env=env)

    def generate_default_virtual_env(self):
        # generate a virtual environment with default parameters
        kwargs = self._get_default_parameters(virtual_env = True)
        print('Generating default virtual environment "{}" with parameters {}'.format(self.env_name, kwargs))
        env = VirtualEnv(kwargs).to(device)
        return EnvWrapper(env=env).to(device)

    def _get_random_parameters(self):
        kwargs = {"env_name": self.env_name,
                  "state_dim": self.state_dim,
                  "action_dim": self.action_dim}
        for key, value in self.env_config.items():
            if isinstance(value, list):
                kwargs[key] = np.random.uniform(low=float(value[0]), high=float(value[2]))
            else:
                kwargs[key] = value
        return kwargs

    def _get_default_parameters(self, virtual_env):
        kwargs = {"env_name": self.env_name}
        if virtual_env:
            kwargs["state_dim"] = self.state_dim
            kwargs["action_dim"] = self.action_dim
        for key, value in self.env_config.items():
            if isinstance(value, list):
                kwargs[key] = float(value[1])
            else:
                kwargs[key] = value
        return kwargs


if __name__ == "__main__":
    import yaml

    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        gen = EnvFactory(config)
        a = gen.generate_default_real_env()
