import yaml
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.actor_critic import Actor_TD3, Critic_Q
from utils import ReplayBuffer, AverageMeter
from envs.env_factory import EnvFactory
from agents.TD3_mod import TD3_Mod
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TD3(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, config):
        super().__init__()

        agent_name = "td3"
        td3_config = config["agents"][agent_name]

        self.max_action = max_action
        self.gamma = td3_config["gamma"]
        self.tau = td3_config["tau"]
        self.policy_delay = td3_config["policy_delay"]
        self.batch_size = td3_config["batch_size"]
        self.init_episodes = td3_config["init_episodes"]
        self.max_episodes = td3_config["max_episodes"]
        self.rb_size = td3_config["rb_size"]
        self.lr = td3_config["lr"]
        self.same_action_num = td3_config["same_action_num"]
        self.early_out_num = td3_config["early_out_num"]
        self.policy_std = td3_config["policy_std"]
        self.policy_std_clip = td3_config["policy_std_clip"]
        self.render_env = config["render_env"]

        self.actor = Actor_TD3(state_dim, action_dim, max_action, agent_name, config).to(device)
        self.actor_target = Actor_TD3(state_dim, action_dim, max_action, agent_name, config).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1 = Critic_Q(state_dim, action_dim, agent_name, config).to(device)
        self.critic_2 = Critic_Q(state_dim, action_dim, agent_name, config).to(device)
        self.critic_target_1 = Critic_Q(state_dim, action_dim, agent_name, config).to(device)
        self.critic_target_2 = Critic_Q(state_dim, action_dim, agent_name, config).to(device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.reset_optimizer()

        self.total_it = 0


    def train(self, env, mod = None):
        replay_buffer = ReplayBuffer(env.get_state_dim(), env.get_action_dim(), max_size=self.rb_size)
        avg_meter_reward = AverageMeter(print_str="Average reward: ")

        # training loop
        for episode in range(self.max_episodes):
            state = env.reset()
            episode_reward = 0

            for t in range(0, env.max_episode_steps(), self.same_action_num):
                # fill replay buffer at beginning
                if episode < self.init_episodes:
                    action = env.get_random_action()
                else:
                    action = (self.actor(state.to(device)).cpu() +
                              torch.randn_like(action) * self.policy_std * self.max_action
                             ).clamp(-self.max_action, self.max_action)

                # live view
                if self.render_env and episode % 5 == 0 and episode >= self.init_episodes:
                    env.render(state, action)

                # modify action
                if episode < self.init_episodes or mod is None:
                    action_mod = action.clone().detach()
                else:
                    action_mod = mod.modify_action(state.to(device), action.to(device))

                # state-action transition
                next_state, reward, done = env.step(action=action_mod, state=state, same_action_num=self.same_action_num)

                if t < env.max_episode_steps() - 1:
                    done_tensor = done
                else:
                    done_tensor = torch.tensor([0], device="cpu", dtype=torch.float32)

                # check
                if any(torch.isinf(state)) or any(torch.isnan(state)):
                    print('early out because state is not finite')
                    break

                replay_buffer.add(state=state, action=action, action_mod=action_mod, next_state=next_state, reward=reward, done=done_tensor)

                state = next_state
                episode_reward += reward

                # train
                if episode > self.init_episodes:
                    self.update(replay_buffer)
                    if mod is not None:
                        mod.update(replay_buffer)
                if done > 0.5:
                    break

            # logging
            avg_meter_reward.update(episode_reward, print_rate=self.early_out_num)

            # quit training if environment is solved
            avg_reward = avg_meter_reward.get_mean(num=self.early_out_num)
            if avg_reward >= env.env.solved_reward and episode > self.init_episodes:
                print("early out after {} episodes with an average reward of {}".format(episode, avg_reward))
                break

        env.close()

        return avg_meter_reward.get_raw_data()


    def update(self, replay_buffer):
        self.total_it += 1

        # Sample replay buffer
        states, actions, _, next_states, rewards, dones = replay_buffer.sample(self.batch_size)

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
            target_Q = rewards + (1 - (dones > 0.5).float()) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1 = self.critic_1(states, actions)
        current_Q2 = self.critic_2(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Compute matching loss

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


    def reset_optimizer(self):
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic_1.parameters()) + list(self.critic_2.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.lr)


    def get_state_dict(self):
        agent_state = {}
        agent_state["td3_actor"] = self.actor.state_dict()
        agent_state["td3_actor_target"] = self.actor_target.state_dict()
        agent_state["td3_critic_1"] = self.critic_1.state_dict()
        agent_state["td3_critic_2"] = self.critic_2.state_dict()
        agent_state["td3_critic_target_1"] = self.critic_target_1.state_dict()
        agent_state["td3_critic_target_2"] = self.critic_target_2.state_dict()
        if self.actor_optimizer:
            agent_state["td3_actor_optimizer"] = self.actor_optimizer.state_dict()
        if self.critic_optimizer:
            agent_state["td3_critic_optimizer"] = self.critic_optimizer.state_dict()
        return agent_state

    def set_state_dict(self, agent_state):
        self.actor.load_state_dict(agent_state["td3_actor"])
        self.actor_target.load_state_dict(agent_state["td3_actor_target"])
        self.critic_1.load_state_dict(agent_state["td3_critic_1"])
        self.critic_2.load_state_dict(agent_state["td3_critic_2"])
        self.critic_target_1.load_state_dict(agent_state["td3_critic_target_1"])
        self.critic_target_2.load_state_dict(agent_state["td3_critic_target_2"])
        if "td3_actor_optimizer" in agent_state.keys():
            self.actor_optimizer.load_state_dict(agent_state["td3_actor_optimizer"])
        if "td3_critic_optimizer" in agent_state.keys():
            self.critic_optimizer.load_state_dict(agent_state["td3_critic_optimizer"])


if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # generate environment
    env_fac = EnvFactory(config)
    real_env = env_fac.generate_default_real_env()

    real_env.seed(seed)

    td3 = TD3(state_dim=real_env.get_state_dim(),
              action_dim=real_env.get_action_dim(),
              max_action=real_env.get_max_action(),
              config=config)
    td3.train(env=real_env)
