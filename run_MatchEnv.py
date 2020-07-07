import numpy as np
import torch
import yaml

from agents.match_env import MatchEnv
from envs.env_factory import EnvFactory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    with open("default_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # generate environment
    env_fac = EnvFactory(config)
    real_env = env_fac.generate_default_real_env()
    virtual_env = env_fac.generate_default_virtual_env()

    me = MatchEnv(config)
    me.train(real_env=real_env, virtual_env=virtual_env, input_seed=0)

