import numpy as np
import torch
import gym
import yaml
from agents.PPO import PPO
from envs.env_factory import EnvFactory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    with open("default_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    seed = config['seed']

    # generate environment
    env_fac = EnvFactory(config)
    env = env_fac.generate_default_real_env()

    # set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    ppo = PPO(state_dim = env.get_state_dim(),
              action_dim = env.get_action_dim(),
              config = config)

    ppo.run(env)