import numpy as np
import torch
import gym
import yaml
from agents.TD3 import TD3
from envs.env_factory import EnvFactory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    with open("default_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    seed = config['seed']

    # generate environment
    env_fac = EnvFactory(config)
    env = env_fac.generate_default_env()
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    # set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    td3 = TD3(state_dim = state_dim,
              action_dim = action_dim,
              config = config)

    td3.run(env)