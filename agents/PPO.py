import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.actor_critic import Actor, Critic_V
from utils import AverageMeter, ReplayBuffer
from envs.env_factory import EnvFactory

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
        self.early_out_num = ppo_config['early_out_num']
        self.same_action_num = ppo_config['same_action_num']

        self.render_env = config["render_env"]

        self.actor = Actor(state_dim, action_dim, agent_name,
                           config).to(device)
        self.critic = Critic_V(state_dim, agent_name, config).to(device)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) +
                                          list(self.critic.parameters()),
                                          lr=ppo_config['lr'],
                                          weight_decay=ppo_config['weight_decay'])

        self.actor_old = Actor(state_dim, action_dim, agent_name,
                               config).to(device)
        self.critic_old = Critic_V(state_dim, agent_name, config).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())


    def train(self, env, input_seed=torch.tensor([0])):
        replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6))
        avg_meter_reward = AverageMeter(print_str='Average reward: ')

        time_step = 0

        # training loop
        for episode in range(self.max_episodes):
            state = env.reset()
            episode_reward = 0

            for t in range(0, env.max_episode_steps(), self.same_action_num):
                with torch.no_grad():
                    time_step += 1

                    # run old policy
                    action = self.actor_old(state.to(device)).cpu()
                    next_state, reward, done = env.step(action=action, state=state, same_action_num=self.same_action_num)

                    # live view
                    if self.render_env and episode % 100 == 0:
                        env.render(state)

                    # check
                    if any(torch.isinf(state)) or any(torch.isnan(state)):
                        print('early out because state is not finite')
                        break

                    replay_buffer.add(state=state, action=action, next_state=next_state, reward=reward, done=done)

                    state = next_state
                    episode_reward += reward

                # train after certain amount of timesteps
                if time_step / env.max_episode_steps() > self.update_episodes:
                    self.update(replay_buffer, env, input_seed)
                    replay_buffer.clear()
                    time_step = 0
                if done:
                    break

            # logging
            avg_meter_reward.update(episode_reward, print_rate=self.early_out_num)

            # quit training if environment is solved
            avg_reward = avg_meter_reward.get_mean(num=self.early_out_num)
            if avg_reward > env.env.solved_reward:
                print("early out after {} episodes with an average reward of {}".format(episode, avg_reward))
                break

        env.close()

        return avg_meter_reward.get_raw_data()


    def update(self, replay_buffer, env, input_seed):
        # Monte Carlo estimate of rewards:
        new_rewards = []
        discounted_reward = 0

        # get states from replay buffer
        states, actions, next_states, rewards, dones = replay_buffer.get_all()

        if env.is_virtual_env():
            # TODO: fix
            states = self.run_env(env, last_states, last_actions, input_seed)

        old_logprobs, _ = self.actor_old.evaluate(states, actions)
        old_logprobs = old_logprobs.detach()

        #calculate rewards
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            new_rewards.insert(0, discounted_reward)

        # normalize advantage function
        new_rewards = torch.FloatTensor(new_rewards).to(device)
        new_rewards = (new_rewards - new_rewards.mean()) / (new_rewards.std() + 1e-5)

        # optimize policy for ppo_epochs:
        for it in range(self.ppo_epochs):
            # evaluate old actions and values :
            logprobs, dist_entropy = self.actor.evaluate(
                states, actions)
            state_values = self.critic(states).squeeze()

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss
            advantages = (new_rewards - state_values).detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                1 + self.eps_clip) * advantages
            loss = - torch.min(surr1, surr2) \
                   + self.vf_coef * F.mse_loss(state_values, new_rewards) \
                   - self.ent_coef * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())


    def get_state_dict(self):
        agent_state = {}
        agent_state['ppo_actor'] = self.actor.state_dict()
        agent_state['ppo_actor_old'] = self.actor_target.state_dict()
        agent_state['ppo_critic'] = self.critic_1.state_dict()
        agent_state['ppo_critic_old'] = self.critic_2.state_dict()
        agent_state['ppo_optimizer'] = self.critic_target_1.state_dict()
        return agent_state

    def set_state_dict(self, agent_state):
        self.actor.load_state_dict(agent_state['ppo_actor'])
        self.actor_old.load_state_dict(agent_state['ppo_actor_old'])
        self.critic.load_state_dict(agent_state['ppo_critic'])
        self.critic_old.load_state_dict(agent_state['ppo_critic_old'])
        self.optimizer.load_state_dict(agent_state['ppo_optimizer'])


if __name__ == "__main__":
    with open("../default_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # generate environment
    env_fac = EnvFactory(config)
    env = env_fac.generate_default_real_env()

    env.seed(seed)

    ppo = PPO(state_dim=env.get_state_dim(),
              action_dim=env.get_action_dim(),
              config=config)
    a = ppo.state_dict()
    ppo.train(env)

