import copy
import torch
import torch.nn as nn
import numpy as np
from agents.TD3 import TD3
from agents.PPO import PPO
from envs.env_generator import EnvGenerator


class REPTILE(nn.Module):
    def __init__(self, config):
        super(REPTILE, self).__init__()

        reptile_config = config['agents']['reptile']
        self.max_iterations = reptile_config['max_iterations']
        self.step_size = reptile_config['step_size']
        self.agent_name = reptile_config['agent_name']

        self.env_generator = EnvGenerator(config)
        self.agent = self.agent_factory(config)

    def run(self):
        for it in range(self.max_iterations):
            old_state_dict = copy.deepcopy(self.agent.state_dict())
            env = self.env_generator.generate_real_env()
            self.agent.run(env)
            new_state_dict = self.agent.state_dict()

            # print('old')
            # print(old_state_dict['actor.action_std'])
            # print(new_state_dict['actor.action_std'])
            # print(self.agent.state_dict()['actor.action_std'])

            for key,value in new_state_dict.items():
                new_state_dict[key] = old_state_dict[key] \
                                      + (new_state_dict[key] - old_state_dict[key]) * self.step_size

            # print('new')
            # print(old_state_dict['actor.action_std'])
            # print(new_state_dict['actor.action_std'])
            # print(self.agent.state_dict()['actor.action_std'])
            #
            # self.agent.load_state_dict(new_state_dict)
            #
            # print('newer')
            # print(old_state_dict['actor.action_std'])
            # print(new_state_dict['actor.action_std'])
            # print(self.agent.state_dict()['actor.action_std'])


    def agent_factory(self, config):
        dummy_env = self.env_generator.generate_default_env()
        state_dim = dummy_env.observation_space.shape[0]
        action_dim = dummy_env.action_space.shape[0]

        if self.agent_name == 'TD3':
            return TD3(state_dim, action_dim, config)
        elif self.agent_name == 'PPO':
            return PPO(state_dim, action_dim, config)
        else:
            raise NotImplementedError('Unknownn RL agent')





