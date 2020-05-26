from utils import Actor, Critic_V
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import gym
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPO:
    def __init__(self,
                 state_dim,
                 action_dim,
                 config):

        ppo_config = config['ppo']

        self.gamma = ppo_config['gamma']
        self.vf_coef = ppo_config['vf_coef']
        self.ent_coef = ppo_config['ent_coef']
        self.eps_clip = ppo_config['eps_clip']
        self.ppo_epochs = ppo_config['ppo_epochs']

        self.actor = Actor(state_dim, action_dim, config).to(device)
        self.critic = Critic_V(state_dim, config).to(device)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) +
                                          list(self.critic.parameters()), lr=ppo_config['lr'])

        self.actor_old = Actor(state_dim, action_dim, config).to(device)
        self.critic_old = Critic_V(state_dim, config).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor_old(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0

        # get states from replay buffer
        old_states, old_actions, _, old_rewards, old_dones = replay_buffer.get_all()
        old_logprobs, _ = self.actor_old.evaluate(old_states, old_actions)
        old_logprobs = old_logprobs.detach()

        #calculate rewards
        for reward, done in zip(reversed(old_rewards), reversed(old_dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # normalize advantage function
        rewards = torch.FloatTensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # optimize policy for ppo_epochs:
        for it in range(self.ppo_epochs):
            # evaluate old actions and values :
            logprobs, dist_entropy = self.actor.evaluate(old_states, old_actions)
            state_values = self.critic(old_states)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = - torch.min(surr1, surr2) \
                   + self.vf_coef * F.mse_loss(state_values, rewards) \
                   - self.ent_coef * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())
