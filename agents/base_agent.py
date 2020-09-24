import time
import torch
import torch.nn as nn
from utils import ReplayBuffer, AverageMeter, time_is_up, env_solved


class BaseAgent(nn.Module):
    def __init__(self, agent_name, state_dim, action_dim, config):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        agent_config = config["agents"][agent_name]
        self.train_episodes = agent_config["train_episodes"]
        self.test_episodes = agent_config["test_episodes"]
        self.init_episodes = agent_config["init_episodes"]
        self.batch_size = agent_config["batch_size"]
        self.same_action_num = agent_config["same_action_num"]
        self.print_rate = agent_config["print_rate"]
        self.rb_size = agent_config["rb_size"]
        self.early_out_num = agent_config["early_out_num"]
        self.early_out_virtual_diff = agent_config["early_out_virtual_diff"]

        self.render_env = config["render_env"]
        self.device = config["device"]


    def train(self, env, time_remaining):
        time_start = time.time()

        sd = 1 if env.has_discrete_state_space() else self.state_dim
        ad = 1 if env.has_discrete_action_space() else self.action_dim
        replay_buffer = ReplayBuffer(state_dim=sd, action_dim=ad, device=self.device, max_size=self.rb_size)

        avg_meter_reward = AverageMeter(print_str="Average reward: ")

        # training loop
        for episode in range(self.train_episodes):
            # early out if timeout
            if time_is_up(avg_meter_reward=avg_meter_reward,
                          max_episodes=self.train_episodes,
                          time_elapsed=time.time() - time_start,
                          time_remaining=time_remaining):
                break

            self.update_parameters_per_episode(episode=episode)

            state = env.reset()
            episode_reward = 0

            for t in range(0, env.max_episode_steps(), self.same_action_num):
                action = self.select_train_action(state=state, env=env, episode=episode)

                # live view
                if self.render_env and episode % 10 == 0:
                    env.render()

                # state-action transition
                next_state, reward, done = env.step(action=action, same_action_num=self.same_action_num)
                replay_buffer.add(state=state, action=action, next_state=next_state, reward=reward, done=done)
                state = next_state
                episode_reward += reward

                # train
                if episode > self.init_episodes:
                    self.learn(replay_buffer=replay_buffer, env=env)

                if done > 0.5:
                    break

            # logging
            avg_meter_reward.update(episode_reward, print_rate=self.print_rate)

            # quit training if environment is solved
            if env_solved(agent=self, env=env, avg_meter_reward=avg_meter_reward, episode=episode):
                break

        env.close()

        return avg_meter_reward.get_raw_data()


    def test(self, env, time_remaining=1e9):
        with torch.no_grad():
            time_start = time.time()

            avg_meter_reward = AverageMeter(print_str="Average reward: ")

            # training loop
            for episode in range(self.test_episodes):
                # early out if timeout
                if time_is_up(avg_meter_reward=avg_meter_reward,
                              max_episodes=self.test_episodes,
                              time_elapsed=time.time() - time_start,
                              time_remaining=time_remaining):
                    break

                state = env.reset()
                episode_reward = 0

                for t in range(0, env.max_episode_steps(), self.same_action_num):
                    action = self.select_test_action(state)

                    # live view
                    if self.render_env and episode % 10 == 0:
                        env.render()

                    # state-action transition
                    next_state, reward, done = env.step(action=action, same_action_num=self.same_action_num)
                    state = next_state
                    episode_reward += reward

                    if done > 0.5:
                        break

                # logging
                avg_meter_reward.update(episode_reward, print_rate=self.print_rate)

                # quit training if environment is solved
                if env_solved(agent=self, env=env, avg_meter_reward=avg_meter_reward, episode=episode):
                    break

            env.close()

        return avg_meter_reward.get_raw_data()
