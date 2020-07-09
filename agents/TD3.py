import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agents.match_env import match_loss
from models.actor_critic import Actor, Critic_Q
from utils import ReplayBuffer, AverageMeter, print_abs_param_sum
from envs.env_factory import EnvFactory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TD3(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super().__init__()

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
        self.lr = td3_config["lr"]
        self.weight_decay = td3_config["weight_decay"]
        self.optim_env_with_actor = td3_config["optim_env_with_actor"]
        self.optim_env_with_critic = td3_config["optim_env_with_critic"]
        self.early_out_num = td3_config["early_out_num"]
        self.match_weight_actor = td3_config["match_weight_actor"]
        self.match_weight_critic = td3_config["match_weight_critic"]
        self.match_batch_size = td3_config["match_batch_size"]

        self.render_env = config["render_env"]

        self.actor = Actor(state_dim, action_dim, agent_name, config).to(device)
        self.actor_target = Actor(state_dim, action_dim, agent_name, config).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1 = Critic_Q(state_dim, action_dim, agent_name, config).to(device)
        self.critic_2 = Critic_Q(state_dim, action_dim, agent_name, config).to(device)
        self.critic_target_1 = Critic_Q(state_dim, action_dim, agent_name, config).to(
            device
        )
        self.critic_target_2 = Critic_Q(state_dim, action_dim, agent_name, config).to(
            device
        )
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = None
        self.critic_optimizer = None

        self.total_it = 0

    def train(self, env, match_env=None, input_seed=0):
        # env=virtual_env, match_env=real_env, input_seed given: Train on variable virtual env
        # env=virtual_env, input_seed given: Train on fixed virtual env
        # env=real_env: Train on real env

        replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, self.rb_size)
        avg_meter_reward = AverageMeter(print_str="Average reward: ")

        self.init_optimizer(env, match_env)

        time_step = 0
        input_seed = torch.tensor([input_seed], device="cpu", dtype=torch.float32)

        # training loop
        for episode in range(self.max_episodes):
            state = env.reset()
            last_action = None
            last_state = None
            episode_reward = 0

            # print(self.actor.net._modules['0'].weight.sum())
            # print(env.env.base._modules['0'].weight.sum())

            for t in range(env.max_episode_steps()):
                time_step += 1

                # required to make resampling of actions (for the autograd bw graph) deterministic
                actor_seed = abs(int(sum(state) * 1e9))
                torch.manual_seed(actor_seed)

                # fill replay buffer at beginning
                if episode < self.init_episodes:
                    action = env.get_random_action()
                else:
                    action = self.actor(state.to(device)).cpu()

                # live view
                if (
                    self.render_env
                    and episode % 10 == 0
                    and episode > self.init_episodes
                ):
                    env.render(state)

                # state-action transition
                next_state, reward, done = env.step(action, state, input_seed)

                if t < env.max_episode_steps() - 1:
                    done_tensor = done
                else:
                    done_tensor = torch.tensor([0], device="cpu", dtype=torch.float32)

                if last_state is not None and last_action is not None:
                    replay_buffer.add(
                        last_state,
                        last_action,
                        state,
                        action,
                        next_state,
                        reward,
                        done_tensor,
                        input_seed,
                        self.actor.action_std.data,
                    )

                last_state = state
                state = next_state
                last_action = action

                episode_reward += reward

                # train
                if episode > self.init_episodes:
                    self.update(replay_buffer, env, match_env)
                if done:
                    break

            # logging
            avg_meter_reward.update(episode_reward, print_rate=self.early_out_num)

            # quit training if environment is solved
            avg_reward = avg_meter_reward.get_mean(num=self.early_out_num)
            if avg_reward > env.env.solved_reward:
                print(
                    "early out after {} episodes with an average reward of {}".format(
                        episode, avg_reward
                    )
                )
                break

        env.close()

        return avg_meter_reward.get_raw_data()

    def update(self, replay_buffer, env, match_env=None):
        match_virtual_env = env.is_virtual_env() and match_env is not None
        self.total_it += 1

        # Sample replay buffer
        (
            last_states,
            last_actions,
            states,
            actions,
            next_states,
            rewards,
            dones,
            input_seeds,
            action_std,
        ) = replay_buffer.sample(self.batch_size)

        if match_virtual_env:
            last_states.requires_grad = True
            last_actions.requires_grad = True
            input_seeds.requires_grad = True

            states, _, _ = self.run_env(env, last_states, last_actions, input_seeds)
            _, rewards, dones = self.run_env(env, states, actions, input_seeds)

        with torch.no_grad():
            # Select action according to policy and add clipped noise, no_grad since target will be copied
            next_actions = self.actor_target(next_states)

            # Compute the target Q value
            target_Q1 = self.critic_target_1(next_states, next_actions)
            target_Q2 = self.critic_target_2(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        actions = self.actor.reconstruct_autograd(states, action_std)

        # Get current Q estimates
        current_Q1 = self.critic_1(states, actions)
        current_Q2 = self.critic_2(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Compute matching loss
        m_loss = 0
        if match_virtual_env and self.optim_env_with_critic:
            m_loss = match_loss(
                real_env=match_env,
                virtual_env=env,
                input_seed=input_seeds[0],
                batch_size=self.match_batch_size,
            )
            m_loss *= self.match_weight_critic

        # if self.total_it % 100 == 0:
        #     print('----')
        #     print(critic_loss)
        #     print(m_loss)

        critic_loss += m_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            if match_virtual_env:
                last_states.requires_grad = True
                last_actions.requires_grad = True
                input_seeds.requires_grad = True

                states, _, _ = self.run_env(env, last_states, last_actions, input_seeds)
                _, rewards, dones = self.run_env(env, states, actions, input_seeds)

            # Compute actor loss
            # todo: check algorithm 1 in original paper; has additional multiplicative term here
            actor_loss = (-self.critic_1(states, self.actor(states))).mean()

            # Compute matching loss
            m_loss = 0
            if match_virtual_env and self.optim_env_with_actor:
                m_loss = match_loss(
                    real_env=match_env,
                    virtual_env=env,
                    input_seed=input_seeds[0],
                    batch_size=self.match_batch_size,
                )
                m_loss *= self.match_weight_actor

            # if self.total_it % 100 == 0:
            #     print('----')
            #     print(actor_loss)
            #     print(m_loss)

            actor_loss += m_loss

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic_1.parameters(), self.critic_target_1.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.critic_2.parameters(), self.critic_target_2.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def init_optimizer(self, env, match_env):
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic_1.parameters()) + list(
            self.critic_2.parameters()
        )
        if env.is_virtual_env() and match_env is not None:
            if self.optim_env_with_actor:
                actor_params += list(env.parameters())
            if self.optim_env_with_critic:
                critic_params += list(env.parameters())
        self.actor_optimizer = torch.optim.Adam(
            actor_params, lr=self.lr, weight_decay=self.weight_decay
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_params, lr=self.lr, weight_decay=self.weight_decay
        )

    def run_env(self, env, last_state, last_action, input_seed):
        # enable gradient computation
        state, reward, done = env.step(last_action, last_state, input_seed)
        state = state.to(device)  # wtf?
        reward = reward.to(device)
        done = done.to(device)

        return state, reward, done

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
        print("td3 set state dict")
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

    # generate environment
    env_fac = EnvFactory(config)
    # env = env_fac.generate_default_real_env()
    env = env_fac.generate_interpolate_real_env(1)

    # set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    td3 = TD3(
        state_dim=env.get_state_dim(), action_dim=env.get_action_dim(), config=config
    )

    td3.train(env)
