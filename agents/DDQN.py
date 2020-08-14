import yaml
import torch
import torch.nn as nn
import gym
import numpy as np
import math
#from models.actor_critic import Critic_DQN
#from utils import ReplayBuffer, AverageMeter
#from envs.env_factory import EnvFactory
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, num_inputs, actions_dim):
        super(DQN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, actions_dim)
        )

    def forward(self, x):
        return self.nn(x)

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, s0, a, r, s1, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s0[None, :], a, r, s1[None, :], done))

    def sample(self, batch_size):
        s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(s0), a, r, np.concatenate(s1), done

    def size(self):
        return len(self.buffer)

class DDQN(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super().__init__()

        agent_name = "ddqn"
        ddqn_config = config["agents"][agent_name]

        self.max_episodes = ddqn_config["max_episodes"]
        self.init_episodes = ddqn_config["init_episodes"]
        self.batch_size = ddqn_config["batch_size"]
        self.same_action_num = ddqn_config["same_action_num"]
        self.gamma = ddqn_config["gamma"]
        self.lr = ddqn_config["lr"]
        self.tau = ddqn_config["tau"]
        self.eps = ddqn_config["eps_init"]
        self.eps_init = ddqn_config["eps_init"]
        self.eps_min = ddqn_config["eps_min"]
        self.eps_decay = ddqn_config["eps_decay"]
        self.rb_size = ddqn_config["rb_size"]
        self.early_out_num = ddqn_config["early_out_num"]
        self.render_env = config["render_env"]

        #self.model = Critic_DQN(state_dim, action_dim, agent_name, config).to(device)
        #self.model_target = Critic_DQN(state_dim, action_dim, agent_name, config).to(device)
        self.model = DQN(state_dim, action_dim).to(device)
        self.model_target = DQN(state_dim, action_dim).to(device)
        self.model_target.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #self.reset_optimizer()

        self.epsilon_by_frame = lambda frame_idx: self.eps_min + (self.eps_init - self.eps_min) * math.exp(
            -1. * frame_idx / self.eps_decay)
        self.use_cuda = True
        self.it = 0


    def act(self, state, epsilon=None):
        if epsilon is None: epsilon = self.eps_min
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            if self.use_cuda:
                state = state.cuda()
            q_value = self.model.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(2)
        return action


    def train(self, env):
        buffer = ReplayBuffer(self.rb_size)

        losses = []
        all_rewards = []
        episode_reward = 0
        ep_num = 0

        state = env.reset()
        for fr in range(1,100000):
            epsilon = self.epsilon_by_frame(fr)
            action = self.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            loss = 0
            if buffer.size() > self.batch_size:
                loss = self.learning(buffer, fr)
                losses.append(loss)

            if fr % 200 == 0:
                print("frames: %5d, reward: %5f, loss: %4f episode: %4d" % (fr, np.mean(all_rewards[-10:]), loss, ep_num))

            if done:
                state = env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
                ep_num += 1

    def learning(self, buffer, fr):
        s0, a, r, s1, done = buffer.sample(self.batch_size)

        s0 = torch.tensor(s0, dtype=torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        if self.use_cuda:
            s0 = s0.cuda()
            s1 = s1.cuda()
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()

        q_values = self.model(s0).cuda()
        next_q_values = self.model(s1).cuda()
        next_q_state_values = self.model_target(s1).cuda()

        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.gamma * next_q_value * (1 - done)
        # Notice that detach the expected_q_value
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if fr % 100 == 0:
            self.model_target.load_state_dict(self.model.state_dict())

        return loss.item()


    # def train(self, env, mod = None):
    #     replay_buffer = ReplayBuffer(env.get_state_dim(), 1, max_size=self.rb_size)
    #     avg_meter_reward = AverageMeter(print_str="Average reward: ")
    #     avg_meter_eps = AverageMeter(print_str="Average eps: ")
    #
    #     # training loop
    #     for episode in range(self.max_episodes):
    #         state = env.reset()
    #         episode_reward = 0
    #
    #         for t in range(0, env.max_episode_steps(), self.same_action_num):
    #             if np.random.randn() < self.eps:
    #                 action = env.get_random_action()
    #             else:
    #                 qvals = self.model(state.to(device))
    #                 action = torch.argmax(qvals).unsqueeze(0).detach()
    #
    #             # live view
    #             if self.render_env and episode % 10 == 0:
    #                 env.render()
    #
    #             # state-action transition
    #             next_state, reward, done = env.step(action=action, state=state, same_action_num=self.same_action_num)
    #             replay_buffer.add(state=state, action=action, action_mod=action.clone(), next_state=next_state, reward=reward, done=done)
    #
    #             state = next_state
    #             episode_reward += reward
    #
    #             # train
    #             if episode > self.init_episodes:
    #                 self.update(replay_buffer)
    #
    #             if done > 0.5:
    #                 break
    #
    #         # logging
    #         avg_meter_reward.update(episode_reward, print_rate=self.early_out_num)
    #         avg_meter_eps.update(self.eps, print_rate=self.early_out_num*10)
    #
    #         # update eps
    #         self.eps *= self.eps_decay
    #         self.eps = max(self.eps, self.eps_min)
    #
    #         # quit training if environment is solved
    #         avg_reward = avg_meter_reward.get_mean(num=self.early_out_num)
    #         if avg_reward >= env.env.solved_reward:
    #             print("early out after {} episodes with an average reward of {}".format(episode, avg_reward))
    #             break
    #
    #     env.close()
    #
    #     return avg_meter_reward.get_raw_data()


    # def update(self, replay_buffer):
    #     self.it += 1
    #
    #     states, actions, _, next_states, rewards, dones = replay_buffer.sample(self.batch_size)
    #
    #     states = states.squeeze()
    #     actions = actions.squeeze()
    #     next_states = next_states.squeeze()
    #     rewards = rewards.squeeze()
    #     dones = dones.squeeze()
    #
    #     # print(states.shape)
    #     # print(actions.shape)
    #     # print(next_states.shape)
    #     # print(rewards.shape)
    #     # print(dones.shape)
    #
    #     # next_Q = self.model(next_states)
    #     # next_Q_target = self.model_target(next_states)
    #     # max_next_Q = next_Q_target.gather(1, next_Q.max(1)[1].unsqueeze(1))
    #     # expected_Q = rewards + (1 - dones) * self.gamma * max_next_Q
    #     #
    #     # curr_Q = self.model(states).gather(1, actions.long())
    #     #
    #     # # print('---')
    #     # # print(actions.shape)
    #     # # print(max_next_Q.shape)
    #     # # print(expected_Q.shape)
    #     # # print(curr_Q.shape)
    #     #
    #     # loss = F.mse_loss(curr_Q, expected_Q.detach())
    #
    #     q_values = self.model(states)
    #     next_q_values = self.model(next_states)
    #     next_q_state_values = self.model_target(next_states)
    #
    #     q_value = q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    #     next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
    #     expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)
    #     # Notice that detach the expected_q_value
    #     loss = (q_value - expected_q_value.detach()).pow(2).mean()
    #
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     # target network update
    #     if self.it % self.tau == 0:
    #         self.model_target.load_state_dict(self.model.state_dict())
    #
    #     # for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
    #     #     target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
    #
    #
    # def reset_optimizer(self):
    #     self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    #
    #
    # def get_state_dict(self):
    #     agent_state = {}
    #     agent_state["ddqn_model"] = self.model.state_dict()
    #     agent_state["ddqn_model_target"] = self.model_target.state_dict()
    #     if self.optimizer:
    #         agent_state["ddqn_optimizer"] = self.optimizer.state_dict()
    #
    #     return agent_state
    #
    # def set_state_dict(self, agent_state):
    #     self.model.load_state_dict(agent_state["ddqn_model"])
    #     self.model_target.load_state_dict(agent_state["ddqn_model_target"])
    #     if "ddqn_optimizer" in agent_state.keys():
    #         self.optimizer.load_state_dict(agent_state["ddqn_optimizer"])


if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # seed = config["seed"]
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # generate environment
    # env_fac = EnvFactory(config)
    # real_env = env_fac.generate_default_real_env()

    # real_env.seed(seed)

    env = gym.make('CartPole-v0')

    # ddqn = DDQN(state_dim=real_env.get_state_dim(),
    #             action_dim=real_env.get_action_dim(),
    #             config=config)

    ddqn = DDQN(state_dim=4,
                action_dim=2,
                config=config)

    ddqn.train(env=env)
    #ddqn.train(env=real_env)
