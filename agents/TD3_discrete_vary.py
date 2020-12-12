import copy
import random
import time

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from agents.base_agent import BaseAgent
from envs.env_factory import EnvFactory
from models.actor_critic import Actor_TD3_discrete, Critic_Q


class TD3_discrete_vary(BaseAgent):
    def __init__(self, env, min_action, max_action, config):
        self.agent_name = 'td3_discrete_vary'

        if config["agents"]["td3_discrete_vary"]["vary_hp"]:
            config_mod = copy.deepcopy(config)
            config_mod = self.vary_hyperparameters(config_mod)
        else:
            print("--- configs not varied ---")
            config_mod = config

        super().__init__(agent_name=self.agent_name, env=env, config=config_mod)

        # todo: change
        self.action_dim = 2  # due to API

        td3_config = config_mod["agents"][self.agent_name]

        self.max_action = max_action
        self.min_action = min_action
        self.batch_size = td3_config["batch_size"]
        self.rb_size = td3_config["rb_size"]
        self.gamma = td3_config["gamma"]
        self.tau = td3_config["tau"]
        self.policy_delay = td3_config["policy_delay"]
        self.lr = td3_config["lr"]
        self.action_std = td3_config["action_std"]
        self.policy_std = td3_config["policy_std"]
        self.policy_std_clip = td3_config["policy_std_clip"]

        self.actor = Actor_TD3_discrete(self.state_dim, self.action_dim, max_action, self.agent_name, config_mod).to(self.device)
        self.actor_target = Actor_TD3_discrete(self.state_dim, self.action_dim, max_action, self.agent_name, config_mod).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1 = Critic_Q(self.state_dim, self.action_dim, self.agent_name, config_mod).to(self.device)
        self.critic_2 = Critic_Q(self.state_dim, self.action_dim, self.agent_name, config_mod).to(self.device)
        self.critic_target_1 = Critic_Q(self.state_dim, self.action_dim, self.agent_name, config_mod).to(self.device)
        self.critic_target_2 = Critic_Q(self.state_dim, self.action_dim, self.agent_name, config_mod).to(self.device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.reset_optimizer()

        self.total_it = 0

    def learn(self, replay_buffer, env, episode):
        self.total_it += 1

        # Sample replay buffer
        states, actions, next_states, rewards, dones = replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise, no_grad since target will be copied
            noise = (torch.randn_like(actions) * self.policy_std
                    ).clamp(-self.policy_std_clip, self.policy_std_clip)
            next_actions = self.actor_target(next_states) + noise

            # Compute the target Q value
            target_Q1 = self.critic_target_1(next_states, next_actions)
            target_Q2 = self.critic_target_2(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q
            # target_Q = rewards + self.gamma * target_Q

        # Get current Q estimates
        current_Q1 = self.critic_1(states, actions)
        current_Q2 = self.critic_2(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss
            # todo: check algorithm 1 in original paper; has additional multiplicative term here
            actor_loss = (-self.critic_1(states, self.actor(states))).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def vary_hyperparameters(self, config_mod):

        lr = config_mod['agents'][self.agent_name]['lr']
        batch_size = config_mod['agents'][self.agent_name]['batch_size']
        hidden_size = config_mod['agents'][self.agent_name]['hidden_size']
        hidden_layer = config_mod['agents'][self.agent_name]['hidden_layer']

        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr', lower=lr / 3, upper=lr * 3, log=True, default_value=lr))
        cs.add_hyperparameter(
                CSH.UniformIntegerHyperparameter(name='batch_size', lower=int(batch_size / 3), upper=int(batch_size * 3), log=True,
                                                 default_value=batch_size))
        cs.add_hyperparameter(
                CSH.UniformIntegerHyperparameter(name='hidden_size', lower=int(hidden_size / 3), upper=int(hidden_size * 3), log=True,
                                                 default_value=hidden_size))
        cs.add_hyperparameter(
                CSH.UniformIntegerHyperparameter(name='hidden_layer', lower=hidden_layer - 1, upper=hidden_layer + 1, log=False,
                                                 default_value=hidden_layer))

        config = cs.sample_configuration()

        print(f"sampled config: "
              f"lr: {config['lr']}, "
              f"batch_size: {config['batch_size']}, "
              f"hidden_size: {config['hidden_size']}, "
              f"hidden_layer: {config['hidden_layer']}"
              )

        config_mod['agents'][self.agent_name]['lr'] = config['lr']
        config_mod['agents'][self.agent_name]['batch_size'] = config['batch_size']
        config_mod['agents'][self.agent_name]['hidden_size'] = config['hidden_size']
        config_mod['agents'][self.agent_name]['hidden_layer'] = config['hidden_layer']

        print(config_mod['agents'][self.agent_name])
        return config_mod

    def select_train_action(self, state, env, episode):
        if episode < self.init_episodes:
            action = env.get_random_action()
            diag = torch.eye(self.action_dim)
            return diag[action.long()]  # for CartPole: 0 -> [1,0], 1 -> [0,1]
        else:
            action = self.actor(state).to(self.device).cpu()
            return action + (torch.randn(self.action_dim) * self.action_std)

    def select_test_action(self, state, env):
        action = self.actor(state).to(self.device).cpu()
        return action + (torch.randn(self.action_dim) * self.action_std)

    def reset_optimizer(self):
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic_1.parameters()) + list(self.critic_2.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.lr)


if __name__ == "__main__":
    with open("../default_config_cartpole.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # print("turning off config sampling")
    # config['agents']['td3_discrete_vary']['vary_hp'] = False

    random.seed(int(time.time()))
    np.random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))
    torch.cuda.manual_seed_all(int(time.time()))

    # generate environment
    env_fac = EnvFactory(config)
    # virt_env = env_fac.generate_virtual_env()
    real_env = env_fac.generate_real_env()
    # reward_env = env_fac.generate_reward_env()
    td3 = TD3_discrete_vary(env=real_env, min_action=real_env.get_min_action(), max_action=real_env.get_max_action(), config=config)
    t1 = time.time()
    td3.train(env=real_env)
    print(time.time() - t1)
    # td3.train(env=virt_env, time_remaining=5)
