import yaml
import copy
import torch
import torch.nn as nn
import numpy as np
from agents.agent_utils import test, print_stats
from envs.env_factory import EnvFactory
from agents.agent_utils import select_agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reptile_train_agent_serial(agent, mod, env, step_size):
    old_state_dict = copy.deepcopy(agent.state_dict())
    agent.train(env=env, mod=mod)
    new_state_dict = copy.deepcopy(agent.state_dict())

    reptile_update_state_dict_serial(agent=agent,
                                     old_state_dict=old_state_dict,
                                     new_state_dict=new_state_dict,
                                     step_size=step_size)


def reptile_train_agent_parallel(agent, mod, envs, step_size):
    old_state_dict = copy.deepcopy(agent.state_dict())
    new_state_dicts = []

    for env in envs:
        agent.load_state_dict(copy.deepcopy(old_state_dict))
        agent.train(env=env, mod=mod)
        new_state_dicts.append(copy.deepcopy(agent.state_dict()))

    reptile_update_state_dict_parallel(agent=agent,
                                       old_state_dict=old_state_dict,
                                       new_state_dicts=new_state_dicts,
                                       step_size=step_size)


def reptile_update_state_dict_serial(agent, old_state_dict, new_state_dict, step_size):
    agent_state_dict = agent.state_dict()
    for key, value in new_state_dict.items():
        agent_state_dict[key] = old_state_dict[key] + (new_state_dict[key] - old_state_dict[key]) * step_size


def reptile_update_state_dict_parallel(agent, old_state_dict, new_state_dicts, step_size):
    n = len(new_state_dicts)
    agent.load_state_dict(copy.deepcopy(old_state_dict))
    agent_state_dict = agent.state_dict()
    for new_state_dict in new_state_dicts:
        for key, value in new_state_dict.items():
            agent_state_dict[key] += (new_state_dict[key] - old_state_dict[key]) * step_size/n


# todo fabio: refactor
class REPTILE(nn.Module):
    def __init__(self, config):
        super().__init__()

        reptile_config = config["agents"]["reptile"]
        self.max_iterations = reptile_config["max_iterations"]
        self.step_size = reptile_config["step_size"]
        self.parallel_update = reptile_config["parallel_update"]
        self.env_num = reptile_config["env_num"]

        agent_name = reptile_config["agent_name"]
        self.env_factory = EnvFactory(config)
        self.agent = select_agent(config, agent_name)

        self.envs = []
        for i in range(self.env_num):
            self.envs.append(self.env_factory.generate_random_real_env())

    def train(self):
        for it in range(self.max_iterations):
            print('-- REPTILE iteration {} --'.format(it))
            if self.parallel_update:
                reptile_train_agent_parallel(agent=self.agent, mod=None, envs=self.envs, step_size=self.step_size)
            else:
                for env in self.envs:
                    reptile_train_agent_serial(agent=self.agent, mod=None, env=env, step_size=self.step_size)
                    print_stats(self.agent)

if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # set seeds
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    reptile = REPTILE(config)
    reptile.train()
    result = test(agent=reptile.agent,
                  env_factory=reptile.env_factory,
                  config=reptile.config)
    print(result)

