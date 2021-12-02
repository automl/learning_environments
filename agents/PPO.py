import statistics

import torch
import torch.nn.functional as F
import yaml

from agents.base_agent import BaseAgent
from envs.env_factory import EnvFactory
from models.actor_critic import Actor_PPO, Critic_V
from models.icm_baseline import ICM
from agents.utils import AverageMeter, ReplayBuffer

import logging

logger = logging.getLogger(__name__)

class PPO(BaseAgent):
    def __init__(self, env, config, icm=False):
        agent_name = 'ppo'
        super().__init__(agent_name=agent_name, env=env, config=config)

        ppo_config = config['agents'][agent_name]

        self.gamma = ppo_config['gamma']
        self.vf_coef = ppo_config['vf_coef']
        self.ent_coef = ppo_config['ent_coef']
        self.eps_clip = ppo_config['eps_clip']
        self.ppo_epochs = ppo_config['ppo_epochs']
        self.update_episodes = ppo_config['update_episodes']
        self.early_out_num = ppo_config['early_out_num']
        self.same_action_num = ppo_config['same_action_num']

        self.ppo_config = ppo_config

        self.render_env = config["render_env"]
        self.device = config["device"]

        self.actor = Actor_PPO(self.state_dim, self.action_dim, agent_name, config).to(self.device)
        self.critic = Critic_V(self.state_dim, agent_name, config).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) +
                                          list(self.critic.parameters()),
                                          lr=ppo_config['lr'])

        self.actor_old = Actor_PPO(self.state_dim, self.action_dim, agent_name, config).to(self.device)
        self.critic_old = Critic_V(self.state_dim, agent_name, config).to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        self.icm = None
        if icm:
            icm_config = config["agents"]["icm"]
            self.icm_lr = icm_config["lr"]
            self.icm_beta = icm_config["beta"]
            self.icm_eta = icm_config["eta"]
            self.icm_feature_dim = icm_config["feature_dim"]
            self.icm_hidden_dim = icm_config["hidden_size"]
            self.icm = ICM(state_dim=self.state_dim,
                           action_dim=self.action_dim,
                           has_discrete_actions=env.has_discrete_action_space(),
                           learning_rate=self.icm_lr,
                           beta=self.icm_beta,
                           eta=self.icm_eta,
                           feature_dim=self.icm_feature_dim,
                           hidden_size=self.icm_hidden_dim,
                           device=self.device)

    def train(self, env, time_remaining=1e9, test_env=None):

        sd = 1 if env.has_discrete_state_space() else self.state_dim
        ad = 1 if env.has_discrete_action_space() else self.action_dim
        replay_buffer = ReplayBuffer(state_dim=sd, action_dim=ad, device=self.device, max_size=self.rb_size)

        avg_meter_reward = AverageMeter(print_str="Average reward: ")
        avg_meter_episode_length = AverageMeter(print_str="Average episode length: ")

        env.set_agent_params(same_action_num=self.same_action_num, gamma=self.gamma)

        time_step = 0

        # training loop
        for episode in range(self.train_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0

            for t in range(0, env.max_episode_steps(), self.same_action_num):
                time_step += self.same_action_num

                # run old policy
                action = self.actor_old(state.to(self.device)).cpu()
                next_state, reward, done = env.step(action=action)

                # live view
                if self.render_env and episode % 100 == 0:
                    env.render()

                replay_buffer.add(state=state, action=action, next_state=next_state, reward=reward, done=done)
                state = next_state
                episode_reward += reward
                episode_length += self.same_action_num

                # train after certain amount of timesteps
                if time_step / env.max_episode_steps() > self.update_episodes:
                    self.learn(replay_buffer)
                    replay_buffer.clear()
                    time_step = 0
                if done > 0.5:
                    break

            # logging
            avg_meter_episode_length.update(episode_length, print_rate=1e9)

            if test_env is not None:
                avg_reward_test_raw, _, _ = self.test(test_env)
                avg_meter_reward.update(statistics.mean(avg_reward_test_raw), print_rate=self.print_rate)
            else:
                avg_meter_reward.update(episode_reward, print_rate=self.print_rate)

            # quit training if environment is solved
            if episode >= self.init_episodes:
                if test_env is not None:
                    break_env = test_env
                else:
                    break_env = env
                if self.env_solved(env=break_env, avg_meter_reward=avg_meter_reward, episode=episode):
                    logger.info('early out after ' + str(episode) + ' episodes')
                    break

        env.close()

        return avg_meter_reward.get_raw_data(), avg_meter_episode_length.get_raw_data(), {}

    def select_train_action(self, state, env):
        return self.actor_old(state.to(self.device)).cpu()

    def select_test_action(self, state, env):
        return self.actor_old(state.to(self.device)).cpu()

    def learn(self, replay_buffer):
        # Monte Carlo estimate of rewards:
        new_rewards = []
        discounted_reward = 0

        # get states from replay buffer
        states, actions, next_states, rewards, dones = replay_buffer.get_all()

        if self.icm:
            rewards += self.icm.compute_intrinsic_rewards(states, next_states, actions)

        old_logprobs, _ = self.actor_old.evaluate(states, actions)
        old_logprobs = old_logprobs.detach()

        # calculate rewards
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done > 0.5:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            new_rewards.insert(0, discounted_reward)

        # normalize advantage function
        new_rewards = torch.FloatTensor(new_rewards).to(self.device)
        new_rewards = (new_rewards - new_rewards.mean()) / (new_rewards.std() + 1e-5)

        # optimize policy for ppo_epochs:
        for it in range(self.ppo_epochs):
            # evaluate old actions and values :
            logprobs, dist_entropy = self.actor.evaluate(states, actions)
            state_values = self.critic(states).squeeze()

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss
            advantages = (new_rewards - state_values).detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = - torch.min(surr1, surr2) + self.vf_coef * F.mse_loss(state_values, new_rewards) - self.ent_coef * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer.step()

        if self.icm:
            self.icm.train(states, next_states, actions)

        # Copy new weights into old policy:
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())


if __name__ == "__main__":
    with open("../configurations/default_config_halfcheetah.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    config['envs']['HalfCheetah-v3']['solved_reward'] = 10000
    # config['envs']['MountainCarContinuous-v0']['solved_reward'] = 10000

    # generate environment
    env_fac = EnvFactory(config)
    real_env = env_fac.generate_real_env()

    ppo = PPO(env=real_env,
              config=config,
              icm=True)
    ppo.train(env=real_env)
