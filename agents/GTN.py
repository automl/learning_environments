import copy
import torch
import torch.nn as nn
import numpy as np
from agents.TD3 import TD3
from agents.PPO import PPO
from envs.env_factory import EnvFactory


class GTN(nn.Module):
    def __init__(self, config):
        super(GTN, self).__init__()

        reptile_config = config["agents"]["reptile"]
        self.max_iterations = reptile_config["max_iterations"]
        self.step_size = reptile_config["step_size"]
        self.agent_name = reptile_config["agent_name"]

        self.env_factory = EnvFactory(config)
        self.agent = self.agent_factory(config)
        self.seeds = torch.tensor(
            [np.random.random() for _ in range(self.max_iterations)], device="cpu", dtype=torch.float32
        ).unsqueeze(1)

        self.virtual_env = self.env_factory.generate_default_virtual_env()

    def run(self):
        for it in range(self.max_iterations):

            # train on real env for a bit
            old_state_dict_agent_real = copy.deepcopy(self.agent.state_dict())
            real_env = self.env_factory.generate_default_real_env()  # todo: random or default real env?
            print("-- training on real env --")
            self.agent.run(env=real_env)
            self.reptile_update(old_state_dict_agent_real, self.agent)

            # now train on virtual env
            print("-- training on virtual env --")
            old_state_dict_agent_virt = copy.deepcopy(self.agent.state_dict())
            old_state_dict_virtual_env = copy.deepcopy(self.virtual_env.state_dict())

            self.virtual_env.set_seed(seed=self.seeds[it])
            self.agent.run(env=self.virtual_env)
            self.reptile_update(old_state_dict_agent_virt, self.agent)
            self.reptile_update(old_state_dict_virtual_env, self.virtual_env)

    def agent_factory(self, config):
        dummy_env = self.env_factory.generate_default_real_env()
        state_dim = dummy_env.observation_space.shape[0]
        action_dim = dummy_env.action_space.shape[0]

        if self.agent_name == "TD3":
            return TD3(state_dim, action_dim, config)
        elif self.agent_name == "PPO":
            return PPO(state_dim, action_dim, config)
        else:
            raise NotImplementedError("Unknownn RL agent")

    def reptile_update(self, old_state_dict, target):
        new_state_dict = target.state_dict()

        for key, value in new_state_dict.items():
            new_state_dict[key] = old_state_dict[key] + (new_state_dict[key] - old_state_dict[key]) * self.step_size

        target.load_state_dict(new_state_dict)
