import yaml
import time
import torch
import torch.nn.functional as F
from agents.base_agent import BaseAgent
from models.actor_critic import Actor_TD3, Critic_Q
from envs.env_factory import EnvFactory
from utils import ReplayBuffer, AverageMeter


class TD3(BaseAgent):
    def __init__(self, env, max_action, config):
        agent_name = "td3"
        super().__init__(agent_name=agent_name, env=env, config=config)

        td3_config = config["agents"][agent_name]

        self.max_action = max_action
        self.batch_size = td3_config["batch_size"]
        self.rb_size = td3_config["rb_size"]
        self.gamma = td3_config["gamma"]
        self.tau = td3_config["tau"]
        self.policy_delay = td3_config["policy_delay"]
        self.lr = td3_config["lr"]
        self.action_std = td3_config["action_std"]
        self.policy_std = td3_config["policy_std"]
        self.policy_std_clip = td3_config["policy_std_clip"]

        self.actor = Actor_TD3(self.state_dim, self.action_dim, max_action, agent_name, config).to(self.device)
        self.actor_target = Actor_TD3(self.state_dim, self.action_dim, max_action, agent_name, config).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1 = Critic_Q(self.state_dim, self.action_dim, agent_name, config).to(self.device)
        self.critic_2 = Critic_Q(self.state_dim, self.action_dim, agent_name, config).to(self.device)
        self.critic_target_1 = Critic_Q(self.state_dim, self.action_dim, agent_name, config).to(self.device)
        self.critic_target_2 = Critic_Q(self.state_dim, self.action_dim, agent_name, config).to(self.device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.reset_optimizer()

        self.total_it = 0


    def learn(self, replay_buffer, env, episode):
        self.total_it += 1

        # Sample replay buffer
        states, actions, next_states, rewards, dones = replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise, no_grad since target will be copied
            noise = (torch.randn_like(actions) * self.policy_std
                     ).clamp(-self.policy_std_clip, self.policy_std_clip)
            next_actions = (self.actor_target(next_states) + noise
                            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1 = self.critic_target_1(next_states, next_actions)
            target_Q2 = self.critic_target_2(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q
            #target_Q = rewards + self.gamma * target_Q

        # Get current Q estimates
        current_Q1 = self.critic_1(states, actions)
        current_Q2 = self.critic_2(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss
            # todo: check algorithm 1 in original paper; has additional multiplicative term here
            actor_loss = (-self.critic_1(states, self.actor(states))).mean()

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


    def select_train_action(self, state, env, episode):
        if episode < self.init_episodes:
            return env.get_random_action()
        else:
            return (self.actor(state.to(self.device)).cpu() +
                    torch.randn(self.action_dim) * self.action_std * self.max_action
                    ).clamp(-self.max_action, self.max_action)


    def select_test_action(self, state, env):
        return (self.actor(state.to(self.device)).cpu() +
                  torch.randn(self.action_dim) * self.action_std * self.max_action
                  ).clamp(-self.max_action, self.max_action)


    def reset_optimizer(self):
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic_1.parameters()) + list(self.critic_2.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.lr)



if __name__ == "__main__":
    # with open("../default_config_cmc_td3_opt.yaml", "r") as stream:
    #     config = yaml.safe_load(stream)
    with open("../default_config_halfcheetah.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    config["device"] = "cpu"
    config["init_episodes"] = 10

    print(config)
    # generate environment
    env_fac = EnvFactory(config)
    #virt_env = env_fac.generate_virtual_env()
    real_env = env_fac.generate_real_env()
    #reward_env = env_fac.generate_reward_env()
    td3 = TD3(env=real_env,
              max_action=real_env.get_max_action(),
              config=config)

    import numpy as np
    import os
    step_times_per_episode_real_env = {}
    for i in range(5):
        _, _, _, step_times_real_env = td3.train(env=real_env, time_remaining=500)
        step_times_per_episode_real_env[real_env.env.env_name + "_" + str(i)] = {
                "step_times_per_episode_real_env": step_times_real_env,
                "step_times_mean": np.mean(np.concatenate(step_times_real_env)),
                "step_times_std": np.std(np.concatenate(step_times_real_env))
                }

    experiment_name = "evaluate_inference_time_" + "HalfCheetah" + "_mode_2" + "_" + "TD3" + "_device_" + config["device"]
    file_name = os.path.join(os.getcwd(), experiment_name + '.pt')

    save_dict = {}
    save_dict['config'] = config

    save_dict['step_times_per_episode_real_env'] = step_times_per_episode_real_env
    step_times_real_env_lst = [np.concatenate(v["step_times_per_episode_real_env"]) for v in step_times_per_episode_real_env.values()]
    step_times_real_env_lst = np.concatenate(step_times_real_env_lst)

    save_dict['step_times_real_env_mean'] = np.mean(step_times_real_env_lst)
    save_dict['step_times_real_env_std'] = np.std(step_times_real_env_lst)

    print("device:", config["device"])

    print("step_times_real_env_mean {:.6f}".format(save_dict['step_times_real_env_mean']))
    print("step_times_real_env_std {:.6f}".format(save_dict['step_times_real_env_std']))
    print("number of real env step calls: {}".format(step_times_real_env_lst.size))




