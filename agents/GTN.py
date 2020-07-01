import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from agents.agent_utils import select_agent
from agents.match_env import MatchEnv
from agents.REPTILE import reptile_update
from envs.env_factory import EnvFactory
from utils import AverageMeter, print_abs_param_sum

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GTN(nn.Module):
    def __init__(self, config):
        super().__init__()

        gtn_config = config["agents"]["gtn"]
        self.max_iterations = gtn_config["max_iterations"]
        self.match_iterations = gtn_config["match_iterations"]
        self.real_iterations = gtn_config["real_iterations"]
        self.virtual_iterations = gtn_config["virtual_iterations"]
        self.step_size = gtn_config["step_size"]

        agent_name = gtn_config["agent_name"]
        self.agent = select_agent(config, agent_name)
        self.matcher = MatchEnv(config)

        different_envs = gtn_config["different_envs"]
        self.env_factory = EnvFactory(config)
        self.virtual_env = self.env_factory.generate_default_virtual_env()
        self.real_envs = []
        self.input_seeds = []
        if different_envs == 0:
            # generate single default environment with fixed seed
            self.real_envs.append(self.env_factory.generate_default_real_env())
            self.input_seeds.append(1)
        else:
            # generate multiple different real envs with associated seed
            for i in range(different_envs):
                self.real_envs.append(self.env_factory.generate_random_real_env())
                self.input_seeds.append(np.random.random())

        # if os.path.isfile(self.export_path):
        #     self.load_checkpoint()

    def print_stats(self):
        print_abs_param_sum(self.virtual_env, 'VirtualEnv')
        print_abs_param_sum(self.agent.actor, 'Actor')
        print_abs_param_sum(self.agent.critic_1, 'Critic1')
        print_abs_param_sum(self.agent.critic_2, 'Critic2')


    def run(self):
        for it in range(self.max_iterations):
            # if it % 10 == 0:
            #     self.save_checkpoint()

            self.print_stats()

            # map virtual env to real env
            print("-- matching virtual env to real env --")
            for _ in range(self.match_iterations):
                env_id = np.random.randint(len(self.real_envs))
                self.matcher.run(real_env = self.real_envs[env_id],
                                 virtual_env = self.virtual_env,
                                 input_seed = self.input_seeds[env_id])

            self.print_stats()

            # now train on real env
            print("-- training on real env --")
            for _ in range(self.real_iterations):
                env_id = np.random.randint(len(self.real_envs))
                self.reptile_run(env = self.real_envs[env_id])

            self.print_stats()

            # now train on virtual env
            print("-- training on virtual env --")
            for _ in range(self.virtual_iterations):
                env_id = np.random.randint(len(self.real_envs))
                self.reptile_run(env=self.virtual_env,
                                 input_seed=self.input_seeds[env_id])

    def reptile_run(self, env, input_seed=0):
        old_state_dict_agent = copy.deepcopy(self.agent.state_dict())
        if env.is_virtual_env():
            old_state_dict_env = copy.deepcopy(self.virtual_env.state_dict())

        self.agent.run(env=env, input_seed=input_seed)

        reptile_update(target = self.agent,
                       old_state_dict = old_state_dict_agent,
                       step_size = self.step_size)
        if env.is_virtual_env():
            reptile_update(target = env,
                           old_state_dict = old_state_dict_env,
                           step_size = self.step_size)


    def validate(self):
        # calculate after how many steps with a new environment a certain score is achieved
        env = self.env_factory.generate_default_real_env()
        results = self.agent.run(env=env)
        return len(results)


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
