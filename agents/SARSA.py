import yaml
import torch
import time
import random
from agents.base_agent import BaseAgent
from envs.env_factory import EnvFactory
from utils import ReplayBuffer, AverageMeter

class SARSA(BaseAgent):
    def __init__(self, env, config):
        self.agent_name = "sarsa"
        super().__init__(agent_name=self.agent_name, env=env, config=config)

        ql_config = config["agents"][self.agent_name]

        self.batch_size = ql_config["batch_size"]
        self.alpha = ql_config["alpha"]
        self.gamma = ql_config["gamma"]
        self.eps_init = ql_config["eps_init"]
        self.eps_min = ql_config["eps_min"]
        self.eps_decay = ql_config["eps_decay"]
        self.q_table = [[0]*self.action_dim for _ in range(self.state_dim)]

        self.it = 0


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

            delta = reward + self.gamma * self.q_table[next_state][next_action] * (done < 0.5) - self.q_table[state][action]
            self.q_table[state][action] += self.alpha * delta

        replay_buffer.clear()


    def plot_q_function(self, env):
        # m = len(env.env.grid)
        # n = len(env.env.grid[0])
        m = 3
        n = 4

        print('----')
        for i in range(m):
            strng = ''
            for k in range(n):
                strng += ' {:3f}'.format(max(self.q_table[i * n + k]))
            print(strng)


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
    with open("../default_config_gridworld.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    torch.set_num_threads(1)

    # generate environment
    env_fac = EnvFactory(config)
    real_env = env_fac.generate_real_env()
    #virtual_env = env_fac.generate_virtual_env()

    timing = []
    for i in range(100):
        sarsa = SARSA(env=real_env,
                      config=config)
        sarsa.train(env=real_env, time_remaining=500)
        reward_list, _, replay_buffer = sarsa.test(env=real_env, time_remaining=500)
        print(sum(reward_list)/len(reward_list))

