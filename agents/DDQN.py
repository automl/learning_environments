import yaml
import torch
import torch.nn as nn
import gym
import numpy as np
import math
import random
import torch.nn.functional as F
from models.actor_critic import Critic_DQN
from utils import ReplayBuffer, AverageMeter
from envs.env_factory import EnvFactory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDQN(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super().__init__()

        agent_name = "ddqn"
        ddqn_config = config["agents"][agent_name]

        self.max_episodes = ddqn_config["max_episodes"]
        self.init_episodes = ddqn_config["init_episodes"]
        self.batch_size = ddqn_config["batch_size"]
        self.same_action_num = ddqn_config["same_action_num"]
        self.gamma = ddqn_config["gamma"]
        self.lr = ddqn_config["lr"]
        self.tau = ddqn_config["tau"]
        self.eps = ddqn_config["eps_init"]
        self.eps_init = ddqn_config["eps_init"]
        self.eps_min = ddqn_config["eps_min"]
        self.eps_decay = ddqn_config["eps_decay"]
        self.rb_size = ddqn_config["rb_size"]
        self.early_out_num = ddqn_config["early_out_num"]
        self.render_env = config["render_env"]

        self.model = Critic_DQN(state_dim, action_dim, agent_name, config).to(device)
        self.model_target = Critic_DQN(state_dim, action_dim, agent_name, config).to(device)
        self.model_target.load_state_dict(self.model.state_dict())

        self.reset_optimizer()

        self.it = 0


    def update(self, env, mod = None):
        replay_buffer = ReplayBuffer(env.get_state_dim(), 1, max_size=self.rb_size)
        avg_meter_reward = AverageMeter(print_str="Average reward: ")
        avg_meter_eps = AverageMeter(print_str="Average eps: ")

        self.eps = self.eps_init

        # training loop
        for episode in range(self.max_episodes):
            state = env.reset()
            episode_reward = 0

            choose_random = 0

            for t in range(0, env.max_episode_steps(), self.same_action_num):
                if random.random() < self.eps:
                    choose_random += 1
                    action = env.get_random_action()
                else:
                    qvals = self.model(state.to(device))
                    action = torch.argmax(qvals).unsqueeze(0).detach()

                # live view
                if self.render_env and episode % 10 == 0:
                    env.render()

                # state-action transition
                next_state, reward, done = env.step(action=action, state=state, same_action_num=self.same_action_num)
                replay_buffer.add(state=state, action=action, next_state=next_state, reward=reward, done=done)
                state = next_state
                episode_reward += reward

                # train
                if episode > self.init_episodes:
                    self.learn(replay_buffer)

                if done > 0.5:
                    #print(str(action) + " " + str(choose_random/(t+1)))
                    break

            # logging
            avg_meter_reward.update(episode_reward, print_rate=self.early_out_num)
            #avg_meter_eps.update(self.eps, print_rate=self.early_out_num)

            # update eps
            self.eps *= self.eps_decay
            self.eps = max(self.eps, self.eps_min)

            # quit training if environment is solved
            avg_reward = avg_meter_reward.get_mean(num=self.early_out_num)
            if avg_reward >= env.env.solved_reward:
                print("early out after {} episodes with an average reward of {}".format(episode, avg_reward))
                break

        env.close()

        return avg_meter_reward.get_raw_data()


    def learn(self, replay_buffer):
        self.it += 1

        states, actions, next_states, rewards, dones = replay_buffer.sample(self.batch_size)

        states = states.squeeze()
        actions = actions.squeeze()
        next_states = next_states.squeeze()
        rewards = rewards.squeeze()
        dones = dones.squeeze()

        if len(states.shape) == 1:
            states = states.unsqueeze(1)
            next_states = next_states.unsqueeze(1)

        q_values = self.model(states)
        next_q_values = self.model(next_states)
        next_q_state_values = self.model_target(next_states)

        q_value = q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        loss = F.mse_loss(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target network update
        for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        return loss


    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def get_state_dict(self):
        agent_state = {}
        agent_state["ddqn_model"] = self.model.state_dict()
        agent_state["ddqn_model_target"] = self.model_target.state_dict()
        if self.optimizer:
            agent_state["ddqn_optimizer"] = self.optimizer.state_dict()

        return agent_state

    def set_state_dict(self, agent_state):
        self.model.load_state_dict(agent_state["ddqn_model"])
        self.model_target.load_state_dict(agent_state["ddqn_model_target"])
        if "ddqn_optimizer" in agent_state.keys():
            self.optimizer.load_state_dict(agent_state["ddqn_optimizer"])


if __name__ == "__main__":
    with open("../default_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # seed = config["seed"]
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # generate environment
    env_fac = EnvFactory(config)
    real_env = env_fac.generate_default_real_env()
    # real_env.seed(seed)

    ddqn = DDQN(state_dim=real_env.get_state_dim(),
                action_dim=real_env.get_action_dim(),
                config=config)

    ddqn.update(env=real_env)


