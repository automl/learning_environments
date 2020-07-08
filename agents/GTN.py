import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from agents.agent_utils import select_agent
from agents.match_env import MatchEnv
from agents.REPTILE import reptile_update, reptile_train_agent, reptile_match_env
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
        self.match_iterations = gtn_config["match_iterations"]
        self.real_iterations = gtn_config["real_iterations"]
        self.virtual_iterations = gtn_config["virtual_iterations"]
        self.step_size = gtn_config["step_size"]

        agent_name = gtn_config["agent_name"]
        self.agent = select_agent(config, agent_name)
        self.match_env = MatchEnv(config)

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

    def print_stats(self):
        print_abs_param_sum(self.virtual_env, 'VirtualEnv')
        print_abs_param_sum(self.agent.actor, 'Actor')
        print_abs_param_sum(self.agent.critic_1, 'Critic1')
        print_abs_param_sum(self.agent.critic_2, 'Critic2')


    def train(self):
        for it in range(self.max_iterations):
            self.print_stats()

            # map virtual env to real env
            print("-- matching virtual env to real env --")
            for _ in range(self.match_iterations):
                old_state_dict_env = copy.deepcopy(self.virtual_env.state_dict())

                env_id = np.random.randint(len(self.real_envs))
                print('-- with id ' + str(env_id) + ' --')
                reptile_match_env(match_env = self.match_env,
                                  real_env = self.real_envs[env_id],
                                  virtual_env = self.virtual_env,
                                  input_seed = self.input_seeds[env_id],
                                  step_size = self.step_size)

            self.print_stats()

            # now train on virtual env
            print("-- training on virtual env --")
            for _ in range(self.real_iterations):
                env_id = np.random.randint(len(self.real_envs))
                reptile_train_agent(agent = self.agent,
                                    env = self.virtual_env,
                                    input_seed = self.input_seeds[env_id])

            self.print_stats()

            # now train on real env
            print("-- training on real env --")
            for _ in range(self.real_iterations):
                env_id = np.random.randint(len(self.real_envs))
                print('-- with id ' + str(env_id) + ' --')
                reptile_train_agent(agent = self.agent,
                                    env = self.real_envs[env_id],
                                    step_size = self.step_size)

            self.print_stats()

            # now train on virtual env
            print("-- training on both environments --")
            for _ in range(self.virtual_iterations):
                env_id = np.random.randint(len(self.real_envs))
                reptile_train_agent(agent = self.agent,
                                    env=self.virtual_env,
                                    match_env=self.real_envs[env_id],
                                    input_seed=self.input_seeds[env_id],
                                    step_size = self.step_size)

    def test(self):
        # calculate after how many steps with a new environment a certain score is achieved
        env = self.env_factory.generate_default_real_env()
        results = self.agent.run(env=env)
        return len(results)

    def save(self, path):
        # not sure if working
        state = {}
        state['config'] = self.config
        state['agent'] = self.agent.get_state_dict()
        state['virtual_env'] = self.virtual_env.get_state_dict()
        state['input_seeds'] = self.input_seeds
        state['real_envs'] = []
        for real_env in self.real_envs:
            state['real_envs'].append(real_env.get_state_dict())
        torch.save(state, path)

    def load(self, path):
        # not sure if working
        if os.path.isfile(path):
            state = torch.load(self.path)
            self.__init__(state['config'])
            self.agent.set_state_dict(state['agent'])
            self.virtual_env.set_state_dict(state['virtual_env'])
            self.input_seeds = state['input_seeds']
            for i in range(len(self.real_envs)):
                self.real_envs[i].set_state_dict(state['real_envs'][i])
        else:
            raise FileNotFoundError('File not found: ' + str(path))
