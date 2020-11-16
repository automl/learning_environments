import yaml
import torch
import time
import random
from agents.base_agent import BaseAgent
from envs.env_factory import EnvFactory
from utils import ReplayBuffer, AverageMeter

class QL(BaseAgent):
    def __init__(self, env, config):
        agent_name = "ql"
        super().__init__(agent_name=agent_name, env=env, config=config)

        ql_config = config["agents"][agent_name]

        self.alpha = ql_config["alpha"]
        self.gamma = ql_config["gamma"]
        self.eps_init = ql_config["eps_init"]
        self.eps_min = ql_config["eps_min"]
        self.eps_decay = ql_config["eps_decay"]
        self.action_noise = ql_config["action_noise"]
        self.action_noise_decay = ql_config["action_noise_decay"]
        self.q_table = [[0]*self.action_dim for _ in range(self.state_dim)]

        self.it = 0


    def train(self, env, time_remaining=1e9):
        time_start = time.time()

        replay_buffer = ReplayBuffer(state_dim=1, action_dim=1, device=self.device, max_size=int(1e6))

        avg_meter_reward = AverageMeter(print_str="Average reward: ")

        # training loop
        for episode in range(self.train_episodes):
            # early out if timeout
            if self.time_is_up(avg_meter_reward=avg_meter_reward,
                               max_episodes=self.train_episodes,
                               time_elapsed=time.time() - time_start,
                               time_remaining=time_remaining):
                break

            self.update_eps(episode=episode)

            state = env.reset()
            episode_reward = 0

            for t in range(0, env.max_episode_steps(), self.same_action_num):
                action = self.select_train_action(state=state, env=env)

                # live view
                if self.render_env and episode % 10 == 0:
                    env.render()

                # state-action transition
                next_state, reward, done = env.step(action=action,
                                                    same_action_num=self.same_action_num,
                                                    action_noise=self.action_noise)
                replay_buffer.add(state=state, action=action, next_state=next_state, reward=reward, done=done)
                state = next_state
                episode_reward += reward

                # train
                self.learn(replay_buffer=replay_buffer, env=env)

                if done > 0.5:
                    break

            # logging
            avg_meter_reward.update(episode_reward, print_rate=self.print_rate)

            # quit training if environment is solved
            if self.env_solved(env=env, avg_meter_reward=avg_meter_reward, episode=episode):
                break

        env.close()

        # states, _, _, _, _ = replay_buffer.get_all()
        # states = [state.item() for state in states.int()]
        # print(states)

        return avg_meter_reward.get_raw_data(), replay_buffer


    def learn(self, replay_buffer, env):
        self.it += 1

        # if self.it % 5000 == 0:
        #     self.plot_q_function(env)

        state, action, next_state, reward, done = replay_buffer.sample(1)
        state = int(state.item())
        action = int(action.item())
        next_state = int(next_state.item())
        reward = reward.item()
        done = done.item()

        delta = reward + self.gamma * max(self.q_table[next_state]) * (done < 0.5) - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * delta


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


    def select_train_action(self, state, env):
        if random.random() < self.eps:
            action = env.get_random_action()
            return action
        else:
            q_vals = torch.tensor(self.q_table[int(state.item())])
            return torch.argmax(q_vals).unsqueeze(0).detach()


    def select_test_action(self, state, env):
        q_vals = torch.tensor(self.q_table[int(state.item())])
        return torch.argmax(q_vals).unsqueeze(0).detach()


    def update_eps(self, episode):
        if episode == 0:
            self.eps = self.eps_init
        else:
            self.eps *= self.eps_decay
            self.eps = max(self.eps, self.eps_min)


    def get_state_dict(self):
        agent_state = {}
        agent_state["ql_q_table"] = self.q_table
        return agent_state

    def set_state_dict(self, agent_state):
        self.q_table = agent_state["sarsa_q_table"]


if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    torch.set_num_threads(1)

    # generate environment
    env_fac = EnvFactory(config)
    real_env = env_fac.generate_default_real_env()
    virtual_env = env_fac.generate_virtual_env()

    timing = []
    for i in range(100):
        ql = QL(env=real_env,
                config=config)

        #ddqn.train(env=virt_env, time_remaining=50)

        t1 = time.time()
        ql.train(env=real_env, time_remaining=500)
        t2 = time.time()
        timing.append(t2-t1)
        reward_list, replay_buffer = ql.test(env=real_env, time_remaining=500)
        print(sum(reward_list)/len(reward_list))
    print('avg. ' + str(sum(timing)/len(timing)))

