import yaml
import copy
import torch
import torch.nn as nn
import numpy as np
from envs.env_factory import EnvFactory
from agents.agent_utils import select_agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reptile_match_env(env_matcher, real_env, virtual_env, input_seeds, step_size):
    old_state_dict_env = copy.deepcopy(virtual_env.state_dict())
    old_input_seeds = copy.deepcopy(input_seeds)

    env_matcher.train(real_env=real_env, virtual_env=virtual_env, input_seeds=input_seeds)

    reptile_update_state_dict(target=virtual_env, old_state_dict=old_state_dict_env, step_size=step_size)

    # for step_size=1 reptile is disabled
    for i in range(len(old_input_seeds)):
        reptile_update_tensor(target=input_seeds[i], old_tensor=old_input_seeds[i], step_size=step_size)


def reptile_train_agent(agent, env, input_seed=None, step_size=None):
    # env=virtual_env, match_env=real_env, input_seed given: Train on variable virtual env
    # env=virtual_env, input_seed given: Train on fixed virtual env
    # env=real_env: Train on real env

    old_state_dict_agent = copy.deepcopy(agent.state_dict())

    reward_list = agent.train(env=env, input_seed=input_seed)

    reptile_update_state_dict(target=agent, old_state_dict=old_state_dict_agent, step_size=step_size)

    return reward_list


def reptile_update_state_dict(target, old_state_dict, step_size):
    new_state_dict = target.state_dict()
    for key, value in new_state_dict.items():
        new_state_dict[key] = old_state_dict[key] + (new_state_dict[key] - old_state_dict[key]) * step_size


def reptile_update_tensor(target, old_tensor, step_size):
    target.data = old_tensor.data + (target.data - old_tensor.data) * step_size


# todo fabio: refactor
class REPTILE(nn.Module):
    def __init__(self, config):
        super().__init__()

        reptile_config = config["agents"]["reptile"]
        self.max_iterations = reptile_config["max_iterations"]
        self.step_size = reptile_config["step_size"]

        agent_name = reptile_config["agent_name"]
        self.env_factory = EnvFactory(config)
        self.agent = select_agent(config, agent_name)

    def train(self):
        for it in range(self.max_iterations):
            old_state_dict = copy.deepcopy(self.agent.state_dict())

            env = self.env_factory.generate_random_real_env()
            self.agent.train(env=env)

            reptile_update_state_dict(self.agent, old_state_dict, self.step_size)


if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # set seeds
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    reptile = REPTILE(config)
    reptile.train()
