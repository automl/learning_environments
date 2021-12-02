import math
import random

import numpy as np
import torch
import yaml

from agents.base_agent import BaseAgent
from envs.env_factory import EnvFactory

import logging

logger = logging.getLogger(__name__)

class SARSA(BaseAgent):
    def __init__(self, env, config, count_based=False):
        self.agent_name = "sarsa"
        super().__init__(agent_name=self.agent_name, env=env, config=config)

        ql_config = config["agents"][self.agent_name]

        self.batch_size = ql_config["batch_size"]
        self.alpha = ql_config["alpha"]
        self.gamma = ql_config["gamma"]
        self.eps_init = ql_config["eps_init"]
        self.eps_min = ql_config["eps_min"]
        self.eps_decay = ql_config["eps_decay"]
        self.q_table = [[0] * self.action_dim for _ in range(self.state_dim)]

        self.count_based = count_based

        self.it = 0

        if self.count_based:
            self.beta = ql_config["beta"]
            self.visitation_table = np.zeros((self.state_dim, self.action_dim))  # n(s,a)
            self.t_hat = np.zeros((self.state_dim, self.action_dim, self.state_dim))

    def learn(self, replay_buffer, env, episode):
        self.it += 1

        # if self.it % 5000 == 0:
        #     self.plot_q_function(env)

        for _ in range(self.batch_size):
            state, action, next_state, reward, done = replay_buffer.sample(1)
            state = int(state.item())
            action = int(action.item())
            next_action = self.select_train_action(state=next_state, env=env, episode=episode)
            next_state = int(next_state.item())
            next_action = int(next_action.item())
            reward = reward.item()
            done = done.item()

            if self.count_based:
                self.visitation_table[state][action] += 1
                intrinsic_reward = self.beta / (math.sqrt(self.visitation_table[state][action]) + 1e-9)
                reward += intrinsic_reward

            delta = reward + self.gamma * self.q_table[next_state][next_action] * (done < 0.5) - self.q_table[state][action]
            self.q_table[state][action] += self.alpha * delta

        replay_buffer.clear()

    def plot_q_function(self, env):
        # m = len(env.env.grid)
        # n = len(env.env.grid[0])
        m = 3
        n = 4

        logger.info('----')
        for i in range(m):
            strng = ''
            for k in range(n):
                strng += ' {:3f}'.format(max(self.q_table[i * n + k]))
            logger.info(strng)

    def select_train_action(self, state, env, episode):
        if random.random() < self.eps:
            action = env.get_random_action()
            return action
        else:
            q_vals = torch.tensor(self.q_table[int(state.item())])
            return torch.argmax(q_vals).unsqueeze(0).detach()

    def select_test_action(self, state, env):
        q_vals = torch.tensor(self.q_table[int(state.item())])
        return torch.argmax(q_vals).unsqueeze(0).detach()

    def update_parameters_per_episode(self, episode):
        if episode == 0:
            self.eps = self.eps_init
        else:
            self.eps *= self.eps_decay
            self.eps = max(self.eps, self.eps_min)


if __name__ == "__main__":
    with open("../configurations/default_config_gridworld_reward_env.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    torch.set_num_threads(1)

    # generate environment
    env_fac = EnvFactory(config)
    real_env = env_fac.generate_real_env()
    # virtual_env = env_fac.generate_virtual_env()

    timing = []
    reward_list_len = []
    for i in range(20):
        sarsa = SARSA(env=real_env,
                      config=config,
                      count_based=False)
        reward_list_train, episode_length_list_train, _ = sarsa.train(env=real_env, time_remaining=500)
        reward_list, _, replay_buffer = sarsa.test(env=real_env, time_remaining=500)
        reward_list_len.append(len(reward_list_train))
        logger.info(len(reward_list_train))
        logger.info(sum(episode_length_list_train))
