import torch
import torch.nn as nn
import torch.nn.functional as F

from models.actor_critic import Actor, Critic_V
from utils import AverageMeter, ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super().__init__()

        agent_name = 'ppo'

        ppo_config = config['agents'][agent_name]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = ppo_config['gamma']
        self.vf_coef = ppo_config['vf_coef']
        self.ent_coef = ppo_config['ent_coef']
        self.eps_clip = ppo_config['eps_clip']
        self.ppo_epochs = ppo_config['ppo_epochs']
        self.max_episodes = ppo_config['max_episodes']
        self.update_episodes = ppo_config['update_episodes']

        self.actor = Actor(state_dim, action_dim, agent_name,
                           config).to(device)
        self.critic = Critic_V(state_dim, agent_name, config).to(device)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) +
                                          list(self.critic.parameters()),
                                          lr=ppo_config['lr'])

        self.actor_old = Actor(state_dim, action_dim, agent_name,
                               config).to(device)
        self.critic_old = Critic_V(state_dim, agent_name, config).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def run(self, env):
        replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)
        avg_meter_reward = AverageMeter(buffer_size=50,
                                        update_rate=50,
                                        print_str='Average reward: ')

        time_step = 0

        # training loop
        for episode in range(self.max_episodes):
            state = env.reset()
            episode_reward = 0

            for t in range(env.max_episode_steps()):
                time_step += 1

                # run old policy
                action = self.actor_old(state.to(device)).cpu()
                next_state, reward, done = env.step(action)
                replay_buffer.add(state, action, next_state, reward, done)
                state = next_state

                episode_reward += reward

                # train after certain amount of timesteps
                if time_step / env.max_episode_steps() > self.update_episodes:
                    self.train(replay_buffer)
                    replay_buffer.clear()
                    time_step = 0
                if done:
                    break

            # logging
            avg_meter_reward.update(episode_reward)

    def train(self, replay_buffer):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0

        # get states from replay buffer
        old_states, old_actions, _, old_rewards, old_dones = replay_buffer.get_all(
        )
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
            logprobs, dist_entropy = self.actor.evaluate(
                old_states, old_actions)
            state_values = self.critic(old_states).squeeze()

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss
            advantages = (rewards - state_values).detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                1 + self.eps_clip) * advantages
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
