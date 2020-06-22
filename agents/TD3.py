import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.actor_critic import Actor, Critic_Q
from utils import AverageMeter, ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TD3(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(TD3, self).__init__()

        agent_name = "td3"
        td3_config = config["agents"][agent_name]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = td3_config["gamma"]
        self.tau = td3_config["tau"]
        self.policy_delay = td3_config["policy_delay"]
        self.batch_size = td3_config["batch_size"]
        self.init_episodes = td3_config["init_episodes"]
        self.max_episodes = td3_config["max_episodes"]
        self.rb_size = td3_config["rb_size"]

        self.render_env = config["render_env"]

        self.actor = Actor(state_dim, action_dim, agent_name, config).to(device)
        self.actor_target = Actor(state_dim, action_dim, agent_name, config).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=td3_config["lr"])

        self.critic_1 = Critic_Q(state_dim, action_dim, agent_name, config).to(device)
        self.critic_2 = Critic_Q(state_dim, action_dim, agent_name, config).to(device)
        self.critic_target_1 = Critic_Q(state_dim, action_dim, agent_name, config).to(device)
        self.critic_target_2 = Critic_Q(state_dim, action_dim, agent_name, config).to(device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=td3_config["lr"]
        )

        self.total_it = 0

    def run(self, env):
        replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, self.rb_size)
        avg_meter_reward = AverageMeter(buffer_size=10, update_rate=10, print_str="Average reward: ")

        time_step = 0

        # training loop
        for episode in range(self.max_episodes):
            state = env.reset()
            episode_reward = 0

            for t in range(env._max_episode_steps):
                time_step += 1

                print('------')

                # fill replay buffer at beginning
                if episode < self.init_episodes:
                    print('random')
                    action = env.random_action()
                else:
                    print('actor')
                    action = self.actor(state.to(device)).cpu()

                # state-action transition
                if self.render_env:
                    env.render()
                next_state, reward, done = env.step(action)

                if t < env._max_episode_steps - 1:
                    done_tensor = done
                else:
                    done_tensor = torch.tensor([0], device="cpu", dtype=torch.float32)

                print(state)
                print(action)
                print(next_state)
                print(reward)
                print(done_tensor)
                replay_buffer.add(state, action, next_state, reward, done_tensor)

                state = next_state

                episode_reward += reward

                # train
                if episode > self.init_episodes:
                    self.train(replay_buffer)
                if done:
                    break

            # logging
            avg_meter_reward.update(episode_reward)
        env.close()

    def train(self, replay_buffer):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            next_action = self.actor_target(next_state)
            reward = reward

            # Compute the target Q value
            target_Q1 = self.critic_target_1(next_state, next_action)
            target_Q2 = self.critic_target_2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done.int()) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # print(self.actor.net._modules['0'].weight[0][0])

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss
            # todo: check algorithm 1 in original paper; has additional multiplicative term here
            actor_loss = (-self.critic_1(state, self.actor(state))).mean()

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
