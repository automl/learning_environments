import yaml
import copy
import torch
import torch.nn as nn
import numpy as np
from envs.env_factory import EnvFactory
from agents.agent_utils import select_agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reptile_train_agent(agent, env, mod_step_size, step_size):
    old_state_dict_agent = copy.deepcopy(agent.state_dict())
    reward_list, replay_buffer = agent.train(env=env, mod_step_size=mod_step_size)
    reptile_update_state_dict(target=agent, old_state_dict=old_state_dict_agent, step_size=step_size)
    return reward_list, replay_buffer


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
