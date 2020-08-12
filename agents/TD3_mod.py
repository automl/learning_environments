import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.actor_critic import Actor_TD3, Critic_Q
from envs.env_factory import EnvFactory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TD3_Mod(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, config):
        super().__init__()

        agent_name = "td3"
        td3_config = config["agents"][agent_name]

        self.max_action = max_action
        self.gamma = td3_config["gamma"]
        self.tau = td3_config["tau"]
        self.policy_delay = td3_config["policy_delay"]
        self.batch_size = td3_config["batch_size"]
        self.rb_size = td3_config["rb_size"]
        self.lr = td3_config["lr"]
        self.policy_std = td3_config["policy_std"]
        self.policy_std_clip = td3_config["policy_std_clip"]
        self.mod_delay = td3_config["mod_delay"]
        self.mod_type = td3_config["mod_type"]
        self.mod_grad_type = td3_config["mod_grad_type"]
        self.mod_grad_step_size = td3_config["mod_grad_step_size"]
        self.mod_grad_steps = td3_config["mod_grad_steps"]
        self.mod_noise_type = td3_config["mod_noise_type"]
        self.mod_noise_std = td3_config["mod_noise_std"]
        self.mod_mult_const = td3_config["mod_mult_const"]
        self.mod_mult = 1

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


    def modify_action(self, state, action):
        action_mod = action.clone().detach()

        if self.mod_type == 0:
            pass

        elif self.mod_type == 1 or self.mod_type == 2:
            action_mod.requires_grad = True
            step_size = self.mod_grad_step_size / self.mod_grad_steps

            for i in range(self.mod_grad_steps):
                q_val = self.critic_1(state, action_mod)
                q_val.backward()

                if self.mod_grad_type == 1:
                    # normal gradient
                    grad = action_mod.grad
                elif self.mod_grad_type == 2:
                    # normalized gradient
                    grad = action_mod.grad / torch.norm(action_mod.grad)
                elif self.mod_grad_type == 3:
                    # inverse gradient
                    grad = action_mod.grad / torch.norm(action_mod.grad)**2
                else:
                    raise NotImplementedError("Unknownn mod_grad_type: " + str(self.mod_grad_type))

                with torch.no_grad():
                    if self.mod_type == 1:
                        action_mod += grad * step_size * self.mod_mult
                    else:
                        action_mod -= grad * step_size * self.mod_mult
                action_mod.grad.data.zero_()

            action_mod.requires_grad = False

        elif self.mod_type == 3:
            if self.mod_noise_type == 1:
                # normal distribution
                noise = torch.randn_like(action) * self.mod_noise_std
            elif self.mod_noise_type == 2:
                # uniform distribution
                noise = (torch.rand_like(action)-0.5) * 2 * self.mod_noise_std
            else:
                raise NotImplementedError("Unknownn mod_noise_type: " + str(self.mod_noise_type))

            action_mod += noise * self.mod_mult

        else:
            raise NotImplementedError("Unknownn mod_type: " + str(self.mod_type))

        return action_mod


    def update_mod_mult(self):
        if not self.mod_mult_const:
            self.mod_mult = np.random.random() * 2


    def update(self, replay_buffer):
        self.total_it += 1

        if self.total_it % self.mod_delay != 0:
            return

        # Sample replay buffer
        states, _, actions_mod, next_states, rewards, dones = replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise, no_grad since target will be copied
            noise = (torch.randn_like(actions_mod) * self.policy_std
                     ).clamp(-self.policy_std_clip, self.policy_std_clip)
            next_actions = (self.actor_target(next_states) + noise
                            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1 = self.critic_target_1(next_states, next_actions)
            target_Q2 = self.critic_target_2(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - (dones > 0.5).float()) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1 = self.critic_1(states, actions_mod)
        current_Q2 = self.critic_2(states, actions_mod)

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

        agent_state["td3_mod_actor"] = self.actor.state_dict()
        agent_state["td3_mod_actor_target"] = self.actor_target.state_dict()
        agent_state["td3_mod_critic_1"] = self.critic_1.state_dict()
        agent_state["td3_mod_critic_2"] = self.critic_2.state_dict()
        agent_state["td3_mod_critic_target_1"] = self.critic_target_1.state_dict()
        agent_state["td3_mod_critic_target_2"] = self.critic_target_2.state_dict()
        if self.actor_optimizer:
            agent_state["td3_mod_actor_optimizer"] = self.actor_optimizer.state_dict()
        if self.critic_optimizer:
            agent_state["td3_mod_critic_optimizer"] = self.critic_optimizer.state_dict()
        return agent_state

    def set_state_dict(self, agent_state):
        self.actor.load_state_dict(agent_state["td3_mod_actor"])
        self.actor_target.load_state_dict(agent_state["td3_mod_actor_target"])
        self.critic_1.load_state_dict(agent_state["td3_mod_critic_1"])
        self.critic_2.load_state_dict(agent_state["td3_mod_critic_2"])
        self.critic_target_1.load_state_dict(agent_state["td3_mod_critic_target_1"])
        self.critic_target_2.load_state_dict(agent_state["td3_mod_critic_target_2"])
        if "td3_mod_actor_optimizer" in agent_state.keys():
            self.actor_optimizer.load_state_dict(agent_state["td3_mod_actor_optimizer"])
        if "td3_mod_critic_optimizer" in agent_state.keys():
            self.critic_optimizer.load_state_dict(agent_state["td3_mod_critic_optimizer"])

