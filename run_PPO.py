import numpy as np
import torch
import gym
import yaml
from agents.PPO import PPO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    with open("default_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    env_name = config['env_name']
    seed = config['seed']

    # generate environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    ppo = PPO(state_dim = state_dim,
              action_dim = action_dim,
              config = config)

    a = pp
    ppo.run(env)