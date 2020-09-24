import yaml
import torch
import time
import random
import numpy as np
from agents.base_agent import BaseAgent
from envs.env_factory import EnvFactory

class QL(BaseAgent):
    def __init__(self, state_dim, action_dim, config):
        agent_name = "ql"
        super().__init__(agent_name=agent_name, state_dim=state_dim, action_dim=action_dim, config=config)

        ql_config = config["agents"][agent_name]

        self.alpha = ql_config["alpha"]
        self.gamma = ql_config["gamma"]
        self.eps_init = ql_config["eps_init"]
        self.eps_min = ql_config["eps_min"]
        self.eps_decay = ql_config["eps_decay"]
        self.q1_table = [[0]*self.action_dim for _ in range(self.state_dim)]
        self.q2_table = [[0]*self.action_dim for _ in range(self.state_dim)]
        self.e_table = [[0]*self.action_dim for _ in range(self.state_dim)]

        self.it = 0

    def learn(self, replay_buffer, env):
        self.it += 1

        if self.it % 10000 == 0:
            self.plot_q_function(env)

        state, action, next_state, reward, done = replay_buffer.sample(self.batch_size)
        state = int(state.item())
        action = int(action.item())
        next_state = int(next_state.item())
        reward = reward.item()
        done = done.item()

        delta = reward + self.gamma * max(self.q1_table[next_state]) * (done < 0.5) - self.q1_table[state][action]
        self.q1_table[state][action] += self.alpha * delta

        # print('{} {} {} {} {}'.format(state, action, next_state, reward, done))
        # a_dash = int(self.select_train_action(torch.tensor(next_state), env).item())
        # a_star = np.argmax(self.q_table[next_state])
        # if self.q_table[next_state][a_dash] == self.q_table[next_state][a_star]:
        #     a_star = a_dash

        # if random.random() > 0.5:
        #     a1_max = np.argmax(self.q1_table[next_state])
        #     delta = reward + self.gamma * self.q2_table[next_state][a1_max] - self.q1_table[state][action]
        #     self.q1_table[state][action] += self.alpha * delta
        # else:
        #     a2_max = np.argmax(self.q2_table[next_state])
        #     delta = reward + self.gamma * self.q1_table[next_state][a2_max] - self.q2_table[state][action]
        #     self.q2_table[state][action] += self.alpha * delta

        #print('{} {} {} {} {}'.format(state, action, next_state, reward, done))
        # delta = reward + self.gamma * max(self.q1_table[next_state]) * (done < 0.5) - self.q1_table[state][action]
        #
        # self.e_table[state][action] += 1
        # for s in range(self.state_dim):
        #     for a in range(self.action_dim):
        #         self.q1_table[s][a] += self.alpha * delta * self.e_table[s][a]
        #         self.e_table[s][a] *= self.gamma * delta


    def plot_q_function(self, env):
        m = len(env.env.grid)
        n = len(env.env.grid[0])

        for i in range(m):
            strng = ''
            for k in range(n):
                strng += ' {:3f}'.format(max(self.q1_table[i * n + k]))
            print(strng)


    def select_train_action(self, state, env, episode=0):
        if random.random() < self.eps:
            action = env.get_random_action()
            return action
        else:
            q1_vals = torch.tensor(self.q1_table[int(state.item())])
            return torch.argmax(q1_vals).unsqueeze(0).detach()
            # q1_vals = torch.tensor(self.q1_table[int(state.item())])
            # q2_vals = torch.tensor(self.q2_table[int(state.item())])
            # return torch.argmax(q1_vals+q2_vals).unsqueeze(0).detach()


    def select_test_action(self, state):
        qvals = torch.tensor(self.q_table[int(state.item())])
        return torch.argmax(qvals).unsqueeze(0).detach()


    def update_parameters_per_episode(self, episode):
        if episode == 0:
            self.eps = self.eps_init
        else:
            self.eps *= self.eps_decay
            self.eps = max(self.eps, self.eps_min)

        # if episode % 10 == 0:
        #     print(self.eps)


    def get_state_dict(self):
        agent_state = {}
        agent_state["sarsa_q_table"] = self.q_table
        return agent_state

    def set_state_dict(self, agent_state):
        self.q_table = agent_state["sarsa_q_table"]


if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    torch.set_num_threads(1)

    # seed = config["seed"]
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # generate environment
    env_fac = EnvFactory(config)
    virt_env = env_fac.generate_virtual_env()
    real_env = env_fac.generate_default_real_env()

    timing = []
    for i in range(10):
        ql = QL(state_dim=virt_env.get_state_dim(),
                action_dim=virt_env.get_action_dim(),
                config=config)

        #ddqn.train(env=virt_env, time_remaining=50)

        t1 = time.time()
        print('TRAIN')
        ql.train(env=real_env, time_remaining=500)
        t2 = time.time()
        timing.append(t2-t1)
        print(t2-t1)
        #print('TEST')
        #reward_list = ddqn.test(env=real_env, time_remaining=500)
    print('avg. ' + str(sum(timing)/len(timing)))

