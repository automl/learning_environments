import yaml
import random
import torch
import torch.nn as nn
import numpy as np
import os
import copy
from time import time
from agents.TD3 import TD3
from agents.agent_utils import select_agent, select_mod
from agents.REPTILE import reptile_train_agent_serial, reptile_train_agent_parallel
from envs.env_factory import EnvFactory
from utils import print_abs_param_sum

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GTN(nn.Module):
    def __init__(self, config):
        super().__init__()

        # for saving/loading
        self.config = config

        gtn_config = config["agents"]["gtn"]
        self.max_iterations = gtn_config["max_iterations"]
        self.step_size = gtn_config["step_size"]
        self.mod_step_size = gtn_config["mod_step_size"]
        self.mod_mult = gtn_config["mod_mult"]

        self.type = gtn_config["type"]
        self.agent_name = gtn_config["agent_name"]
        self.agent = select_agent(config, self.agent_name)
        self.mod = select_mod(config)

        self.env_factory = EnvFactory(config)
        self.real_env = self.env_factory.generate_default_real_env()

    def print_stats(self):
        print_abs_param_sum(self.agent.actor, "Actor")
        print_abs_param_sum(self.agent.critic_1, "Critic1")
        print_abs_param_sum(self.agent.critic_2, "Critic2")

    def train(self):
        order = []
        timings = []

        for it in range(self.max_iterations):
            print('-- type {} with mod_step_size {} --'.format(self.type, self.mod_step_size))
            self.print_stats()
            t = time()

            if self.type == 1:
                mod_step_sizes = [0,0,0]
            elif self.type == 2:
                mod_step_sizes = [0,
                                  self.mod_step_size * self.mod_mult ** it,
                                  -self.mod_step_size * self.mod_mult ** it]
            else:
                raise ValueError('Case that shoud not happen')

            n = len(mod_step_sizes)
            for mod_step_size in mod_step_sizes:
                reptile_train_agent_serial(agent=self.agent,
                                           mod=self.mod,
                                           env=self.real_env,
                                           mod_step_size=mod_step_size,
                                           step_size=self.step_size/n)
            # reptile_train_agent_parallel(agent=self.agent,
            #                              mod=self.mod,
            #                              env=self.real_env,
            #                              mod_step_sizes=mod_step_sizes,
            #                              step_size=self.step_size)
            order.append(self.type)
            timings.append(int(time()-t))

        self.print_stats()

        return order, timings

    def test(self):
        # generate 10 different deterministic environments with increasing difficulty
        # and check for every environment how many episodes it takes the agent to solve it
        # N.B. we have to reset the state of the agent before every iteration

        # todo future: fine-tuning, then test
        # to avoid problems with wrongly initialized optimizers
        if isinstance(self.agent, TD3):
            self.agent.reset_optimizer()

        mean_episodes_till_solved = 0
        episodes_till_solved = []
        agent_state = copy.deepcopy(self.agent.get_state_dict())

        if self.config['env_name'] == 'HalfCheetah-v2':
            interpolate_vals = [0, 0.03, 0.1, 0.4, 1]
        else:
            interpolate_vals = np.arange(0, 1.01, 0.2)

        for interpolate in interpolate_vals:
            self.agent.set_state_dict(agent_state)
            self.print_stats()
            env = self.env_factory.generate_interpolated_real_env(interpolate)
            reward_list = self.agent.train(env=env, mod=self.mod, mod_step_size=0)
            mean_episodes_till_solved += len(reward_list)
            episodes_till_solved.append(len(reward_list))
            print("episodes till solved: " + str(len(reward_list)))

        self.agent.set_state_dict(agent_state)
        mean_episodes_till_solved /= len(interpolate_vals)

        return mean_episodes_till_solved, episodes_till_solved

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
    gtn.train()
    result = gtn.test()
    print(result)
