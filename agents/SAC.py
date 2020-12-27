import yaml
import time
import torch
import numpy as np
import torch.nn.functional as F
from agents.base_agent import BaseAgent
from models.actor_critic import Actor_SAC, Critic_Q
from envs.env_factory import EnvFactory


class SAC(BaseAgent):
    def __init__(self, env, max_action, config):
        agent_name = 'sac'

        super().__init__(agent_name=agent_name, env=env, config=config)

        sac_config = config["agents"][agent_name]

        self.max_action = max_action
        self.gamma = sac_config["gamma"]
        self.alpha = sac_config["alpha"]
        self.tau = sac_config["tau"]
        self.automatic_entropy_tuning = sac_config["automatic_entropy_tuning"]
        self.rb_size = sac_config["rb_size"]
        self.batch_size = sac_config["batch_size"]
        self.lr = sac_config["lr"]

        self.actor = Actor_SAC(self.state_dim, self.action_dim, max_action, agent_name, config)
        self.critic_1 = Critic_Q(self.state_dim, self.action_dim, agent_name, config).to(self.device)
        self.critic_2 = Critic_Q(self.state_dim, self.action_dim, agent_name, config).to(self.device)
        self.critic_target_1 = Critic_Q(self.state_dim, self.action_dim, agent_name, config).to(self.device)
        self.critic_target_2 = Critic_Q(self.state_dim, self.action_dim, agent_name, config).to(self.device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        if self.automatic_entropy_tuning:
            self.target_entropy = -np.prod((self.action_dim,)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)

        self.reset_optimizer()


    def learn(self, replay_buffer, env, episode):
        states, actions, next_states, rewards, dones = replay_buffer.sample(self.batch_size)
        rewards = rewards.squeeze()
        dones = dones.squeeze()

        # Prediction π(a|s), logπ(a|s), π(a'|s'), logπ(a'|s'), Q1(s,a), Q2(s,a)
        _, pi, log_pi = self.actor(states)
        _, next_pi, next_log_pi = self.actor(next_states)
        q1 = self.critic_1(states, actions).squeeze(1)
        q2 = self.critic_2(states, actions).squeeze(1)

        # Min Double-Q: min(Q1(s,π(a|s)), Q2(s,π(a|s))), min(Q1‾(s',π(a'|s')), Q2‾(s',π(a'|s')))
        min_q_pi = torch.min(self.critic_1(states, pi), self.critic_2(states, pi)).squeeze(1).to(self.device)
        min_q_next_pi = torch.min(self.critic_target_1(next_states, next_pi),
                                  self.critic_target_2(next_states, next_pi)).squeeze(1).to(self.device)

        # Targets for Q regression
        v_backup = min_q_next_pi - self.alpha * next_log_pi
        q_backup = rewards + self.gamma * (1 - dones) * v_backup
        q_backup.to(self.device)

        # SAC losses
        actor_loss = (self.alpha * log_pi - min_q_pi).mean()
        critic1_loss = F.mse_loss(q1, q_backup.detach())
        critic2_loss = F.mse_loss(q2, q_backup.detach())
        critic_loss = critic1_loss + critic2_loss

        # Update policy network parameter
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update two Q-network parameter
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # If automatic entropy tuning is True, update alpha
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # Polyak averaging for target parameter
        for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def select_train_action(self, state, env, episode):
        if episode < self.init_episodes:
            return env.get_random_action()
        else:
            _, action, _ = self.actor(state.to(self.device))
            return action.cpu()


    def select_test_action(self, state, env):
        action, _, _ = self.actor(state.to(self.device))
        return action.cpu()


    def reset_optimizer(self):
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic_1.parameters()) + list(self.critic_2.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.lr)



if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # generate environment
    env_fac = EnvFactory(config)
    #virt_env = env_fac.generate_virtual_env()
    real_env= env_fac.generate_real_env()
    #reward_env = env_fac.generate_reward_env()
    td3 = SAC(env=real_env,
              max_action=real_env.get_max_action(),
              config=config)
    t1 = time.time()
    td3.train(env=real_env, time_remaining=1200)
    print(time.time()-t1)
    td3.test(env=real_env, time_remaining=1200)
    print(time.time()-t1)
    #td3.train(env=virt_env, time_remaining=5)
