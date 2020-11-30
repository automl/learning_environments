import os
import torch
import time
import random
import numpy as np
import torch.nn.functional as F
import statistics
import matplotlib.pyplot as plt
from agents.DDQN import DDQN
from agents.base_agent import BaseAgent
from models.actor_critic import Critic_DQN
from envs.env_factory import EnvFactory
from utils import ReplayBuffer, AverageMeter, to_one_hot_encoding


class DDQN_noise(BaseAgent):
    def __init__(self, env, config):
        agent_name = "ddqn"

        super().__init__(agent_name=agent_name, env=env, config=config)

        ddqn_config = config["agents"][agent_name]

        self.init_episodes = ddqn_config["init_episodes"]
        self.batch_size = ddqn_config["batch_size"]
        self.rb_size = ddqn_config["rb_size"]
        self.gamma = ddqn_config["gamma"]
        self.lr = ddqn_config["lr"]
        self.tau = ddqn_config["tau"]
        self.eps = ddqn_config["eps_init"]
        self.eps_init = ddqn_config["eps_init"]
        self.eps_min = ddqn_config["eps_min"]
        self.eps_decay = ddqn_config["eps_decay"]

        self.model = Critic_DQN(self.state_dim, self.action_dim, agent_name, config).to(self.device)
        self.model_target = Critic_DQN(self.state_dim, self.action_dim, agent_name, config).to(self.device)
        self.model_target.load_state_dict(self.model.state_dict())

        self.reset_optimizer()

        self.it = 0

    def train(self, env, time_remaining=1e9, noise_type=None, noise_value=None):
        time_start = time.time()

        sd = 1 if env.has_discrete_state_space() else self.state_dim
        ad = 1 if env.has_discrete_action_space() else self.action_dim
        replay_buffer = ReplayBuffer(state_dim=sd, action_dim=ad, device=self.device, max_size=self.rb_size)

        avg_meter_reward = AverageMeter(print_str="Average reward: ")

        # training loop
        for episode in range(self.train_episodes):
            # early out if timeout
            if self.time_is_up(avg_meter_reward=avg_meter_reward,
                               max_episodes=self.train_episodes,
                               time_elapsed=time.time() - time_start,
                               time_remaining=time_remaining):
                break

            self.update_parameters_per_episode(episode=episode)

            state = env.reset()
            episode_reward = 0

            for t in range(0, env.max_episode_steps(), self.same_action_num):
                action = self.select_train_action(state=state, env=env)

                # live view
                if self.render_env and episode % 10 == 0:
                    env.render()

                # state-action transition
                next_state, reward, done = env.step(action=action, same_action_num=self.same_action_num)
                next_state, reward = self.apply_noise(next_state=next_state, reward=reward, noise_type=noise_type, noise_value=noise_value)
                replay_buffer.add(state=state, action=action, next_state=next_state, reward=reward, done=done)
                state = next_state
                episode_reward += reward

                # train
                if episode > self.init_episodes:
                    self.learn(replay_buffer=replay_buffer, env=env)

                if done > 0.5:
                    break

            # logging
            avg_meter_reward.update(episode_reward, print_rate=self.print_rate)

            # quit training if environment is solved
            if self.env_solved(env=env, avg_meter_reward=avg_meter_reward, episode=episode):
                break

        env.close()

        return avg_meter_reward.get_raw_data(), replay_buffer

    def apply_noise(self, next_state, reward, noise_type, noise_value):
        if noise_type == None or noise_value == None:
            return next_state, reward
        elif noise_type < len(next_state):
            next_state[noise_type] += torch.randn(1).item()*noise_value
        else:
            reward += torch.randn(1).item()*noise_value

        return next_state, reward

    def learn(self, replay_buffer, env):
        self.it += 1

        states, actions, next_states, rewards, dones = replay_buffer.sample(self.batch_size)

        states = states.squeeze()
        actions = actions.squeeze()
        next_states = next_states.squeeze()
        rewards = rewards.squeeze()
        dones = dones.squeeze()

        if env.has_discrete_state_space():
            states = to_one_hot_encoding(states, self.state_dim)
            next_states = to_one_hot_encoding(next_states, self.state_dim)

        q_values = self.model(states)
        next_q_values = self.model(next_states)
        next_q_state_values = self.model_target(next_states)

        q_value = q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        loss = F.mse_loss(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target network update
        for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        return loss

    def select_train_action(self, state, env):
        if random.random() < self.eps:
            return env.get_random_action()
        else:
            if env.has_discrete_state_space():
                state = to_one_hot_encoding(state, self.state_dim)
            qvals = self.model(state.to(self.device))
            return  torch.argmax(qvals).unsqueeze(0).detach()

    def select_test_action(self, state, env):
        if env.has_discrete_state_space():
            state = to_one_hot_encoding(state, self.state_dim)
        qvals = self.model(state.to(self.device))
        return torch.argmax(qvals).unsqueeze(0).detach()

    def update_parameters_per_episode(self, episode):
        if episode == 0:
            self.eps = self.eps_init
        else:
            self.eps *= self.eps_decay
            self.eps = max(self.eps, self.eps_min)

    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)



def load_envs_and_config(dir, model_file_name):
    file_path = os.path.join(dir, model_file_name)
    save_dict = torch.load(file_path)
    config = save_dict['config']
    # config['envs']['CartPole-v0']['solved_reward'] = 195
    # config['envs']['CartPole-v0']['max_steps'] = 200
    env_factory = EnvFactory(config=config)
    virtual_env = env_factory.generate_virtual_env()
    virtual_env.load_state_dict(save_dict['model'])
    real_env = env_factory.generate_real_env()

    return virtual_env, real_env, config


def calc_noisy_reward(virtual_env, real_env, config, reward_file_name):
    reward_list = [[] for i in range(5)]

    for noise_type in range(5):
        for noise_value in np.logspace(-2, 1.5, num=50):
            reward_sum = []
            train_length = []
            for i in range(100):
                print('{} {} {}'.format(noise_type, noise_value, i))
                agent = DDQN_noise(env=real_env, config=config)
                print('train')
                reward_train, _ = agent.train(env=virtual_env, noise_type=noise_type, noise_value=noise_value)
                print('test')
                reward_test, _ = agent.test(env=real_env)
                reward_sum += reward_test
                train_length.append(len(reward_train))
            reward_avg = statistics.mean(reward_sum)
            train_length_avg = statistics.mean(train_length)

            reward_list[noise_type].append((noise_value, reward_avg, train_length_avg))

    data = {}
    data['reward_list'] = reward_list

    torch.save(data, reward_file_name)


def calc_reference_deviation(virtual_env, real_env, config):

    state_reward_concat = None

    for i in range(10):
        agent = DDQN(env=real_env, config=config)
        _, replay_buffer_train = agent.train(env=virtual_env)

        states, _, _, rewards, _ = replay_buffer_train.get_all()
        state_reward = torch.cat((states, rewards), 1)

        if state_reward_concat == None:
            state_reward_concat = state_reward
        else:
            state_reward_concat = torch.cat((state_reward_concat, state_reward), 0)

        print(state_reward_concat.shape)
        print(torch.std(state_reward_concat, dim=0))

    return torch.std(state_reward_concat, dim=0).item()


def plot_threshold(std_dev):
    plt.figure(dpi=600, figsize=(6,3))

    data = torch.load(reward_file_name)
    reward_list = data['reward_list']

    for i in range(len(reward_list)):
        xs = [elem[0]/std_dev[i] for elem in reward_list[0]]
        ys = [elem[1] for elem in reward_list[i]]
        plt.plot(xs, ys)

    plt.legend(['cart position', 'cart velocity', 'pole angle', 'pole ang. vel.', 'reward'])
    plt.xscale('log')
    plt.xlabel('relative noise intensity')
    plt.ylabel('avg. cumulative reward')
    plt.savefig('cartpole_threshold.svg', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    #dir = '/home/dingsda/master_thesis/learning_environments/results/GTN_models_CartPole-v0'
    dir = '/home/nierhoff/master_thesis/learning_environments/results/GTN_models_CartPole-v0_old'
    model_file_name = 'CartPole-v0_24_I8EZDI.pt'
    reward_file_name = 'GTNC_visualize_cartpole_threshold_rewards_100.pt'

    virtual_env, real_env, config = load_envs_and_config(dir=dir, model_file_name=model_file_name)
    config['device'] = 'cuda'
    config['agents']['ddqn']['print_rate'] = 10
    config['agents']['ddqn']['test_episodes'] = 10
    config['agents']['ddqn']['train_episodes'] = 100

    calc_noisy_reward(virtual_env=virtual_env, real_env=real_env, config=config, reward_file_name=reward_file_name)
    #std_dev = calc_reference_deviation(virtual_env=virtual_env, real_env=real_env, config=config, reward_file_name=reward_file_name)
    #std_dev = [0.0361, 0.0653, 0.0722, 0.0465, 0.0976]
    #plot_threshold(std_dev)





