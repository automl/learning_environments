import yaml
import random
import torch
import torch.nn as nn
import numpy as np
import os
from time import time
from agents.agent_utils import select_agent, test, print_stats
from agents.REPTILE import reptile_update_agent_serial
from envs.env_factory import EnvFactory

class GTN(nn.Module):
    def __init__(self, config):
        super().__init__()

        # for saving/loading
        self.config = config

        gtn_config = config["agents"]["gtn"]
        self.max_iterations = gtn_config["max_iterations"]
        self.step_size = gtn_config["step_size"]

        self.agent_name = gtn_config["agent_name"]
        self.agent = select_agent(config, self.agent_name)

        self.env_factory = EnvFactory(config)
        self.real_env = self.env_factory.generate_default_real_env()

    def update(self):
        order = []
        timings = []

        for it in range(self.max_iterations):
            print_stats(self.agent)
            t = time()

            reptile_update_agent_serial(agent=self.agent,
                                       env=self.real_env,
                                       step_size=self.step_size)

            timings.append(int(time()-t))

        print_stats(self.agent)

        return order, timings

    def save(self, path):
        # not sure if working
        state = {}
        state["config"] = self.config
        state["agent"] = self.agent.get_state_dict()
        state["virtual_env"] = self.virtual_env.get_state_dict()
        state["input_seeds"] = self.input_seeds
        state["real_envs"] = []
        for real_env in self.real_envs:
            state["real_envs"].append(real_env.get_state_dict())
        torch.save(state, path)

    def load(self, path):
        # not sure if working
        if os.path.isfile(path):
            state = torch.load(self.path)
            self.__init__(state["config"])
            self.agent.set_state_dict(state["agent"])
            self.virtual_env.set_state_dict(state["virtual_env"])
            self.input_seeds = state["input_seeds"]
            for i in range(len(self.real_envs)):
                self.real_envs[i].set_state_dict(state["real_envs"][i])
        else:
            raise FileNotFoundError("File not found: " + str(path))


if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # set seeds
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    gtn = GTN(config)
    gtn.update()
    result = test(agent=gtn.agent,
                  env_factory=gtn.env_factory,
                  config=gtn.config)
    print(result)
