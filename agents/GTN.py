import yaml
import torch
import torch.nn as nn
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
        print(gtn_config)
        self.match_iterations = gtn_config["match_iterations"]
        self.max_iterations = gtn_config["max_iterations"]
        self.real_prob = gtn_config["real_prob"]
        self.virtual_prob = gtn_config["virtual_prob"]
        self.both_prob = gtn_config["virtual_prob"]
        self.match_step_size = gtn_config["match_step_size"]
        self.real_step_size = gtn_config["real_step_size"]
        self.virtual_step_size = gtn_config["virtual_step_size"]
        self.both_step_size = gtn_config["both_step_size"]

        agent_name = gtn_config["agent_name"]
        self.agent = select_agent(config, agent_name)
        self.match_env = MatchEnv(config)

        different_envs = gtn_config["different_envs"]
        self.env_factory = EnvFactory(config)
        self.virtual_env = self.env_factory.generate_default_virtual_env()
        self.real_envs = []
        self.input_seeds = []

        # first environment is default environment
        self.real_envs.append(self.env_factory.generate_default_real_env())
        self.input_seeds.append(1)
        # generate multiple different real envs with associated seed
        for i in range(different_envs-1):
            self.real_envs.append(self.env_factory.generate_random_real_env())
            self.input_seeds.append(np.random.random())


    def print_stats(self):
        print_abs_param_sum(self.virtual_env, 'VirtualEnv')
        print_abs_param_sum(self.agent.actor, 'Actor')
        print_abs_param_sum(self.agent.critic_1, 'Critic1')
        print_abs_param_sum(self.agent.critic_2, 'Critic2')


    def train(self):
        self.print_stats()

        # first map virtual env to default real env -> use as starting point for further optimization

        path = 'virtual_env.pt'
        if os.path.isfile(path):
            self.virtual_env.load(path)
        else:
            env_id = 0
            print("-- matching virtual env to real env with id " + str(env_id) + ' --')
            for _ in range(self.match_iterations):
                reptile_match_env(match_env=self.match_env,
                                  real_env=self.real_envs[env_id],
                                  virtual_env=self.virtual_env,
                                  input_seed=self.input_seeds[env_id],
                                  step_size=self.match_step_size)
            self.virtual_env.save(path)

        # randomly determine in each iteration whether
        # - the agent should be trained on a specific real env
        # - the agent should be trained on a fixed virtual env with specific input seed
        # - the agent should be trained on a variable virtual env with specific input seed
        # all conditions has a corresponding probability to be executed in each iteration
        for it in range(self.max_iterations):
            sm = self.real_prob + self.virtual_prob + self.both_prob
            prob = np.random.random() * sm

            self.print_stats()

            env_id = np.random.randint(len(self.real_envs))
            if prob <= self.real_prob:
                print("-- training on real env with id " + str(env_id) + ' --')
                reptile_train_agent(agent = self.agent,
                                    env = self.real_envs[env_id],
                                    step_size = self.real_step_size)

            elif prob <= self.real_prob + self.virtual_prob:
                print("-- training on virtual env with id " + str(env_id) + ' --')
                reptile_train_agent(agent = self.agent,
                                    env = self.virtual_env,
                                    input_seed = self.input_seeds[env_id],
                                    step_size = self.virtual_step_size)

            elif prob <= self.real_prob + self.virtual_prob + self.both_prob:
                print("-- training on both envs with id " + str(env_id) + ' --')
                reptile_train_agent(agent = self.agent,
                                    env=self.virtual_env,
                                    match_env=self.real_envs[env_id],
                                    input_seed=self.input_seeds[env_id],
                                    step_size = self.both_step_size)
            else:
                print('Case that should not happen')


    def test(self):
        # generate 10 different deterministic environments with increasing difficulty
        # and check for every environment how many episodes it takes the agent to solve it
        # N.B. we have to reset the state of the agent before every iteration
        sum_episodes_till_solved = 0
        agent_state = self.agent.get_state_dict()

        for interval in np.arange(0, 1.01, 0.1):
            self.agent.set_state_dict(agent_state)
            env = self.env_factory.generate_interval_real_env(interval)
            reward_list = self.agent.test(env=env)
            sum_episodes_till_solved += len(reward_list)
            print('episodes till solved: ' + str(len(reward_list)))

        self.agent.set_state_dict(agent_state)

        return sum_episodes_till_solved


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


if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # set seeds
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    gtn = GTN(config)
    gtn.train()

