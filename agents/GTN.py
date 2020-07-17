import yaml
import random
import torch
import torch.nn as nn
import numpy as np
import os
import copy
from agents.TD3 import TD3
from agents.agent_utils import select_agent
from agents.env_matcher import EnvMatcher
from agents.REPTILE import reptile_update_state_dict, reptile_train_agent, reptile_match_env
from envs.env_factory import EnvFactory
from utils import AverageMeter, print_abs_param_sum

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GTN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # for saving/loading
        self.config = config

        gtn_config = config["agents"]["gtn"]
        self.max_iterations = gtn_config["max_iterations"]
        self.match_step_size = gtn_config["match_step_size"]
        self.real_step_size = gtn_config["real_step_size"]
        self.virtual_step_size = gtn_config["virtual_step_size"]
        self.both_step_size = gtn_config["both_step_size"]
        self.pretrain_env = gtn_config["pretrain_env"]
        self.pretrain_agent = gtn_config["pretrain_agent"]
        self.type = []
        for i in range(10):
            strng = "type_" + str(i)
            self.type.append(gtn_config[strng])

        agent_name = gtn_config["agent_name"]
        self.agent = select_agent(config, agent_name)
        self.env_matcher = EnvMatcher(config)

        different_envs = gtn_config["different_envs"]
        self.env_factory = EnvFactory(config)
        self.virtual_env = self.env_factory.generate_default_virtual_env()
        self.real_envs = []
        self.input_seeds = []

        # first environment is default environment
        self.real_envs.append(self.env_factory.generate_default_real_env())
        self.input_seeds.append(self.env_factory.generate_default_input_seed())
        # generate multiple different real envs with associated seed
        for i in range(different_envs - 1):
            self.real_envs.append(self.env_factory.generate_random_real_env())
            self.input_seeds.append(self.env_factory.generate_random_input_seed())

    def print_stats(self):
        print_abs_param_sum(self.virtual_env, "VirtualEnv")
        print_abs_param_sum(self.agent.actor, "Actor")
        print_abs_param_sum(self.agent.critic_1, "Critic1")
        print_abs_param_sum(self.agent.critic_2, "Critic2")

    def train(self):
        self.print_stats()

        if self.pretrain_env:
            print("-- matching virtual env to real envs ---")
            reptile_match_env(env_matcher=self.env_matcher,
                              real_envs=self.real_envs,
                              virtual_env=self.virtual_env,
                              input_seeds=self.input_seeds,
                              step_size=self.match_step_size)

        if self.pretrain_agent:
            env_id = 0
            print("-- training on real env with id " + str(env_id) + " --")
            reptile_train_agent(agent=self.agent,
                                env=self.real_envs[env_id],
                                step_size=self.real_step_size)

        order = []
        for it in range(self.max_iterations):
            self.print_stats()

            env_id = np.random.randint(len(self.real_envs))
            if self.type[it] == 1:
                print("-- training on real env with id " + str(env_id) + " --")
                reptile_train_agent(agent=self.agent,
                                    env=self.real_envs[env_id],
                                    step_size=self.real_step_size)

            elif self.type[it] == 2:
                print("-- training on virtual env with id " + str(env_id) + " --")
                reptile_train_agent(agent=self.agent,
                                    env=self.virtual_env,
                                    input_seed=self.input_seeds[env_id],
                                    step_size=self.virtual_step_size)

            elif self.type[it] == 3:
                print("-- training on both envs with id " + str(env_id) + " --")
                reptile_train_agent(agent=self.agent,
                                    env=self.virtual_env,
                                    match_env=self.real_envs[env_id],
                                    input_seed=self.input_seeds[env_id],
                                    step_size=self.both_step_size)

            else:
                print("Case that should not happen")

            order.append(self.type[it])

        return order

    def test(self):
        # generate 10 different deterministic environments with increasing difficulty
        # and check for every environment how many episodes it takes the agent to solve it
        # N.B. we have to reset the state of the agent before every iteration

        # todo future: fine-tuning, then test

        # to avoid problems with wrongly initialized optimizers
        if isinstance(self.agent, TD3):
            env = self.env_factory.generate_default_real_env()
            self.agent.init_optimizer(env=env)

        mean_episodes_till_solved = 0
        agent_state = copy.deepcopy(self.agent.get_state_dict())

        interpolate_vals = np.arange(0, 1.01, 0.1)
        for interpolate in interpolate_vals:
            self.agent.set_state_dict(agent_state)
            # print(self.agent.actor._modules['net']._modules['0'].weight[0][0])
            env = self.env_factory.generate_interpolate_real_env(interpolate)
            reward_list = self.agent.train(env=env)
            mean_episodes_till_solved += len(reward_list)
            print("episodes till solved: " + str(len(reward_list)))

        self.agent.set_state_dict(agent_state)
        mean_episodes_till_solved /= len(interpolate_vals)

        return mean_episodes_till_solved

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    gtn = GTN(config)
    gtn.train()
    result = gtn.test()
    print(result)
