import datetime
import sys
import yaml
import random
import numpy as np
import torch
from copy import deepcopy
from agents.TD3 import TD3
from agents.match_env import MatchEnv
from agents.REPTILE import reptile_train_agent
from envs.env_factory import EnvFactory


if __name__ == "__main__":
    with open("../default_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    env_fac = EnvFactory(config)
    real_env = env_fac.generate_default_real_env()
    virtual_env = env_fac.generate_default_virtual_env()

    td3 = TD3(state_dim=real_env.get_state_dim(),
              action_dim=real_env.get_action_dim(),
              config=config)

    # # first match
    # print("-- matching virtual env to real env --")
    # match_env = MatchEnv(config=config)
    # match_env.train(real_env=real_env,
    #                 virtual_env=virtual_env,
    #                 input_seed=0)
    #
    # path = 'model.pt'
    # virtual_env.save(path)

    path = 'model.pt'
    virtual_env.load(path)

    # then train on virtual env
    print("-- training on virtual env --")
    reptile_train_agent(agent = td3,
                        env = virtual_env,
                        step_size = 0.1)

    # then train on real env
    # ideally the reptile update works and we can train on this environment rather quickly
    print("-- training on real env --")
    reptile_train_agent(agent = td3,
                        env = real_env,
                        step_size = 0.1)


    # path = 'model.pt'
    # virtual_env.load(path)
    #
    # state = virtual_env.reset()
    #
    # for i in range(200):
    #     action = virtual_env.get_random_action()
    #     # state-action transition
    #     state, reward, done = virtual_env.step(action, state)
    #     virtual_env.render(state)
    #



