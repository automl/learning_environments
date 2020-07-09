import yaml
import copy
import torch
import torch.nn as nn
import numpy as np
from envs.env_factory import EnvFactory
from agents.agent_utils import select_agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reptile_match_env(match_env, real_env, virtual_env, input_seed, step_size):
    old_state_dict_env = copy.deepcopy(virtual_env.state_dict())

    match_env.train(real_env=real_env,
                    virtual_env=virtual_env,
                    input_seed=input_seed)

    reptile_update(target=virtual_env,
                   old_state_dict=old_state_dict_env,
                   step_size=step_size)


def reptile_train_agent(agent, env, match_env=None, input_seed=0, step_size=0.1):
    # env=virtual_env, match_env=real_env, input_seed given: Train on variable virtual env
    # env=virtual_env, input_seed given: Train on fixed virtual env
    # env=real_env: Train on real env

    old_state_dict_agent = copy.deepcopy(agent.state_dict())
    if match_env is not None:
        old_state_dict_env = copy.deepcopy(env.state_dict())

    reward_list = agent.train(env=env, match_env=match_env, input_seed=input_seed)

    reptile_update(target = agent,
                   old_state_dict = old_state_dict_agent,
                   step_size = step_size)
    if match_env is not None:
        reptile_update(target = env,
                       old_state_dict = old_state_dict_env,
                       step_size = step_size)

    return reward_list


def reptile_update(target, old_state_dict, step_size):
    new_state_dict = target.state_dict()
    for key, value in new_state_dict.items():
        new_state_dict[key] = old_state_dict[key] + (new_state_dict[key] - old_state_dict[key]) * step_size


class REPTILE(nn.Module):
    def __init__(self, config):
        super().__init__()

        reptile_config = config['agents']['reptile']
        self.max_iterations = reptile_config['max_iterations']
        self.step_size = reptile_config['step_size']

        agent_name = reptile_config['agent_name']
        self.env_factory = EnvFactory(config)
        self.agent = select_agent(config, agent_name)

    def train(self):
        for it in range(self.max_iterations):
            old_state_dict = copy.deepcopy(self.agent.state_dict())

            env = self.env_factory.generate_random_real_env()
            #env = self.env_factory.generate_default_virtual_env().to(device)
            self.agent.train(env=env, input_seed=1)

            reptile_update(self.agent, old_state_dict, self.step_size)


if __name__ == "__main__":
    with open("../default_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # set seeds
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    reptile = REPTILE(config)
    reptile.train()





