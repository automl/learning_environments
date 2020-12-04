import time

import torch
import torch.nn.functional as F
import yaml
import copy

from agents.TD3 import TD3
from agents.base_agent import BaseAgent
from envs.env_factory import EnvFactory
from models.actor_critic import Actor_TD3, Critic_Q
from utils import ReplayBuffer, AverageMeter, to_one_hot_encoding

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

class TD3_discrete_vary(TD3, BaseAgent):
    def __init__(self, env, max_action, config):
        self.agent_name = 'td3_discrete_vary'

        # todo: remove
        config['agents']['td3_discrete_vary']['vary_hp'] = True

        if config["agents"]["td3_discrete_vary"]["vary_hp"]:
            config_mod = copy.deepcopy(config)
            config_mod = self.vary_hyperparameters(config_mod)
        else:
            config_mod = config

        print(config_mod['agents'][self.agent_name]['lr'])
        print(config_mod['agents'][self.agent_name]['batch_size'])
        print(config_mod['agents'][self.agent_name]['hidden_size'])
        print(config_mod['agents'][self.agent_name]['hidden_layer'])

        BaseAgent.__init__(self, agent_name=self.agent_name, env=env, config=config_mod)

        self.action_dim = 1  # due to API

        td3_config = config["agents"][self.agent_name]

        self.max_action = max_action
        self.batch_size = td3_config["batch_size"]
        self.rb_size = td3_config["rb_size"]
        self.gamma = td3_config["gamma"]
        self.tau = td3_config["tau"]
        self.policy_delay = td3_config["policy_delay"]
        self.lr = td3_config["lr"]
        self.action_std = td3_config["action_std"]
        self.policy_std = td3_config["policy_std"]
        self.policy_std_clip = td3_config["policy_std_clip"]

        self.actor = Actor_TD3(self.state_dim, self.action_dim, max_action, self.agent_name, config).to(self.device)
        self.actor_target = Actor_TD3(self.state_dim, self.action_dim, max_action, self.agent_name, config).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1 = Critic_Q(self.state_dim, self.action_dim, self.agent_name, config).to(self.device)
        self.critic_2 = Critic_Q(self.state_dim, self.action_dim, self.agent_name, config).to(self.device)
        self.critic_target_1 = Critic_Q(self.state_dim, self.action_dim, self.agent_name, config).to(self.device)
        self.critic_target_2 = Critic_Q(self.state_dim, self.action_dim, self.agent_name, config).to(self.device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.reset_optimizer()

        self.total_it = 0

    def vary_hyperparameters(self, config_mod):

        lr = config_mod['agents'][self.agent_name]['lr']
        batch_size = config_mod['agents'][self.agent_name]['batch_size']
        hidden_size = config_mod['agents'][self.agent_name]['hidden_size']
        hidden_layer = config_mod['agents'][self.agent_name]['hidden_layer']

        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr', lower=lr/3, upper=lr*3, log=True, default_value=lr))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='batch_size', lower=int(hidden_size/3), upper=int(hidden_size*3), log=True, default_value=batch_size))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='hidden_size', lower=int(hidden_size/3), upper=int(hidden_size*3), log=True, default_value=hidden_size))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='hidden_layer', lower=hidden_layer-1, upper=hidden_layer+1, log=False, default_value=hidden_layer))

        config = cs.sample_configuration()

        print(config_mod['agents'][self.agent_name])
        config_mod['agents'][self.agent_name]['lr'] = config['lr']
        config_mod['agents'][self.agent_name]['batch_size'] = config['batch_size']
        config_mod['agents'][self.agent_name]['hidden_size'] = config['hidden_size']
        config_mod['agents'][self.agent_name]['hidden_layer'] = config['hidden_layer']

        return config_mod

    def select_train_action(self, state, env, episode):
        if episode < self.init_episodes:
            return env.get_random_action()
        else:
            action = torch.round((self.actor(state.to(self.device)).cpu() +
                    torch.randn(self.action_dim) * self.action_std * self.max_action
                    ).clamp(-self.max_action, self.max_action) / 2 + 0.5)
            return action

    def select_test_action(self, state, env):
        return torch.round((self.actor(state.to(self.device)).cpu() +
                    torch.randn(self.action_dim) * self.action_std * self.max_action
                    ).clamp(-self.max_action, self.max_action) / 2 + 0.5)


if __name__ == "__main__":
    with open("../default_config_cartpole.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # generate environment
    env_fac = EnvFactory(config)
    # virt_env = env_fac.generate_virtual_env()
    real_env = env_fac.generate_real_env()
    # reward_env = env_fac.generate_reward_env()
    td3 = TD3_discrete_vary(env=real_env, max_action=real_env.get_max_action(), config=config)
    t1 = time.time()
    td3.train(env=real_env)
    print(time.time() - t1)
    # td3.train(env=virt_env, time_remaining=5)