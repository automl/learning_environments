import copy
import torch
import torch.nn as nn
import numpy as np
import os
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
        self.export_path = config["export_path"]
        self.config = config

        self.env_factory = EnvFactory(config)
        self.agent = self.agent_factory(config)
        self.seeds = torch.tensor(
            [np.random.random() for _ in range(self.max_iterations)], device="cpu", dtype=torch.float32
        ).unsqueeze(1)

        self.virtual_env = self.env_factory.generate_default_virtual_env()

        if os.path.isfile(self.export_path):
            self.load_checkpoint()

    def run(self):
        for it in range(self.max_iterations):

            if it % 10 == 0:
                self.save_checkpoint()

            # train on real env for a bit
            old_state_dict_agent_real = copy.deepcopy(self.agent.state_dict())
            self.real_env = self.env_factory.generate_default_real_env()  # todo: random or default real env?
            print("-- training on real env --")
            self.agent.run(env=self.real_env)
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

    def save_checkpoint(self):
        if self.agent_name == "PPO":
            state_optimizer = {"optimizer": self.agent.optimizer.state_dict()}
        elif self.agent_name == "TD3":
            state_optimizer = {
                "critic_optimizer": self.agent.critic_optimizer.state_dict(),
                "actor_optimizer": self.agent.actor_optimizer.state_dict(),
            }
        state = {
            "agent_state_dict": self.agent.state_dict(),
            "env_factory": self.env_factory,
            "virtual_env_state_dict": self.virtual_env.state_dict(),
            "seeds": self.seeds,
            "config": self.config,  # not loaded
        }

        state = {**state, **state_optimizer}
        torch.save(state, self.export_path)

    def load_checkpoint(self):
        if os.path.isfile(self.export_path):
            state = torch.load(self.export_path)
            self.agent.load_state_dict(state["agent_state_dict"])
            self.env_factory = state["env_factory"]
            self.virtual_env.load_state_dict(state["virtual_env_state_dict"])
            self.seeds = state["seeds"]

            if self.agent_name == "PPO":
                self.agent.optimizer.load_state_dict(state["optimizer"])
            elif self.agent_name == "TD3":
                self.agent.critic_optimizer.load_state_dict(state["critic_optimizer"])
                self.agent.actor_optimizer.load_state_dict(state["actor_optimizer"]),

            print("=> loaded checkpoint '{}'".format(self.export_path))
