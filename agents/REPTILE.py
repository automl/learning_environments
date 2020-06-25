import copy
import torch
import torch.nn as nn
from agents.TD3 import TD3
from agents.PPO import PPO
from envs.env_factory import EnvFactory
from agents.agent_utils import select_agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class REPTILE(nn.Module):
    def __init__(self, config):
        super().__init__()

        reptile_config = config['agents']['reptile']
        self.max_iterations = reptile_config['max_iterations']
        self.step_size = reptile_config['step_size']

        agent_name = reptile_config['agent_name']
        self.env_factory = EnvFactory(config)
        self.agent = select_agent(config, agent_name)

    def run(self):
        for it in range(self.max_iterations):
            old_state_dict = copy.deepcopy(self.agent.state_dict())

            #env = self.env_factory.generate_random_real_env()
            env = self.env_factory.generate_default_virtual_env().to(device)
            self.agent.run(env=env, input_seed=1)
            new_state_dict = self.agent.state_dict()

            for key,value in new_state_dict.items():
                new_state_dict[key] = old_state_dict[key] \
                                      + (new_state_dict[key] - old_state_dict[key]) * self.step_size






