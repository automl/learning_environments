import statistics
import time
import torch
import torch.nn as nn
from agents.utils import AverageMeter, ReplayBuffer


import logging

logger = logging.getLogger(__name__)

class BaseAgent(nn.Module):
    def __init__(self, agent_name, env, config):
        super().__init__()

        self.state_dim = env.get_state_dim()
        self.action_dim = env.get_action_dim()

        agent_config = config["agents"][agent_name]
        self.train_episodes = agent_config["train_episodes"]
        self.test_episodes = agent_config["test_episodes"]
        self.init_episodes = agent_config["init_episodes"]
        self.rb_size = agent_config["rb_size"]
        self.same_action_num = agent_config["same_action_num"]
        self.print_rate = agent_config["print_rate"]
        self.early_out_num = agent_config["early_out_num"]
        self.early_out_virtual_diff = agent_config["early_out_virtual_diff"]

        self.render_env = config["render_env"]
        self.device = config["device"]

        # self.trajectories = []

    def time_is_up(self, avg_meter_reward, avg_meter_episode_length, max_episodes, time_elapsed, time_remaining):
        if time_elapsed > time_remaining:
            logger.info("timeout")
            # fill remaining rewards with minimum reward achieved so far
            if len(avg_meter_reward.get_raw_data()) == 0:
                avg_meter_reward.update(-1e9)
            while len(avg_meter_reward.get_raw_data()) < max_episodes:
                avg_meter_reward.update(min(avg_meter_reward.get_raw_data()), print_rate=1e9)

            # also fill remaining episode lengths with maximum length achieved so far so that avg_meter_reward and avg_meter_episode_length
            # have same length for computing AUC
            if len(avg_meter_episode_length.get_raw_data()) == 0:
                avg_meter_episode_length.update(1e9)
            while len(avg_meter_episode_length.get_raw_data()) < max_episodes:
                avg_meter_episode_length.update(max(avg_meter_episode_length.get_raw_data()), print_rate=1e9)
            return True
        else:
            return False

    def env_solved(self, env, avg_meter_reward, episode):
        avg_reward = avg_meter_reward.get_mean(num=self.early_out_num)
        avg_reward_last = avg_meter_reward.get_mean_last(num=self.early_out_num)
        if env.is_virtual_env():
            if abs(avg_reward - avg_reward_last) / (abs(avg_reward_last) + 1e-9) < self.early_out_virtual_diff and \
                    episode >= self.init_episodes + self.early_out_num:
                logger.info("early out on virtual env after {} episodes with an average reward of {}".format(episode + 1, avg_reward))
                return True
        else:
            if avg_reward >= env.get_solved_reward():
                logger.info("early out on real env after {} episodes with an average reward of {}".format(episode + 1, avg_reward))
                return True

        return False

    def train(self, env, test_env=None, time_remaining=1e9):
        time_start = time.time()

        discretize_action = False

        sd = 1 if env.has_discrete_state_space() else self.state_dim

        # todo: @fabio use "hasattr" and custom function in derived class (see below)
        if env.has_discrete_action_space():
            ad = 1
            # in case of td3_discrete, action_dim=1 does not reflect the required action_dim for the gumbel softmax distribution
            if "td3_discrete" in self.agent_name:
                ad = env.get_action_dim()
                discretize_action = True
        else:
            ad = self.action_dim

        replay_buffer = ReplayBuffer(state_dim=sd, action_dim=ad, device=self.device, max_size=self.rb_size)

        avg_meter_reward = AverageMeter(print_str="Average reward: ")
        avg_meter_episode_length = AverageMeter(print_str="Average episode length: ")

        env.set_agent_params(same_action_num=self.same_action_num, gamma=self.gamma)

        # training loop
        for episode in range(self.train_episodes):
            # early out if timeout
            if self.time_is_up(avg_meter_reward=avg_meter_reward,
                               avg_meter_episode_length=avg_meter_episode_length,
                               max_episodes=self.train_episodes,
                               time_elapsed=time.time() - time_start,
                               time_remaining=time_remaining):
                break

            if hasattr(self, 'update_parameters_per_episode'):
                self.update_parameters_per_episode(episode=episode)

            state = env.reset()
            episode_reward = 0
            episode_length = 0
            for t in range(0, env.max_episode_steps(), self.same_action_num):
                action = self.select_train_action(state=state, env=env, episode=episode)

                # live view
                if self.render_env:
                    env.render()

                # state-action transition
                # required due to gumble softmax in td3 discrete
                # todo @fabio: move into agent-specific select_train_action, do the same for test
                if discretize_action:
                    next_state, reward, done = env.step(action=action.argmax().unsqueeze(0))
                else:
                    next_state, reward, done = env.step(action=action)
                replay_buffer.add(state=state, action=action, next_state=next_state, reward=reward, done=done)

                state = next_state
                episode_reward += reward
                episode_length += self.same_action_num

                # train
                if episode >= self.init_episodes:
                    self.learn(replay_buffer=replay_buffer, env=env, episode=episode)

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

        # todo: use dict to reduce confusions and bugs
        return avg_meter_reward.get_raw_data(), avg_meter_episode_length.get_raw_data(), replay_buffer

    def test(self, env, time_remaining=1e9):
        discretize_action = False

        sd = 1 if env.has_discrete_state_space() else self.state_dim

        if env.has_discrete_action_space():
            ad = 1
            # in case of td3_discrete, action_dim=1 does not reflect the required action_dim for the gumbel softmax distribution
            if self.agent_name == "td3_discrete_vary":
                ad = env.get_action_dim()
                discretize_action = True
        else:
            ad = self.action_dim

        replay_buffer = ReplayBuffer(state_dim=sd, action_dim=ad, device=self.device, max_size=int(1e6))

        env.set_agent_params(same_action_num=self.same_action_num, gamma=self.gamma)

        with torch.no_grad():
            time_start = time.time()

            avg_meter_reward = AverageMeter(print_str="Average reward: ")
            avg_meter_episode_length = AverageMeter(print_str="Average episode length: ")

            # training loop
            for episode in range(self.test_episodes):
                # episode_trajectory = []
                # early out if timeout
                if self.time_is_up(avg_meter_reward=avg_meter_reward,
                                   avg_meter_episode_length=avg_meter_episode_length,
                                   max_episodes=self.test_episodes,
                                   time_elapsed=time.time() - time_start,
                                   time_remaining=time_remaining):
                    break

                state = env.reset()
                episode_reward = 0
                episode_length = 0

                for t in range(0, env.max_episode_steps(), self.same_action_num):
                    action = self.select_test_action(state, env)

                    # live view
                    if self.render_env:
                        env.render()

                    # state-action transition
                    # required due to gumble softmax in td3 discrete
                    if discretize_action:
                        next_state, reward, done = env.step(action=action.argmax().unsqueeze(0))
                    else:
                        next_state, reward, done = env.step(action=action)
                    replay_buffer.add(state=state, action=action, next_state=next_state, reward=reward, done=done)

                    # episode_trajectory.append([state, action, next_state, reward])

                    state = next_state
                    episode_reward += reward
                    episode_length += 1

                    if done > 0.5:
                        break

                # self.trajectories.append(episode_trajectory)

                # logging
                avg_meter_episode_length.update(episode_length, print_rate=1e9)
                avg_meter_reward.update(episode_reward.item(), print_rate=self.print_rate)

            env.close()

        # todo: use dict to reduce confusions and bugs
        return avg_meter_reward.get_raw_data(), avg_meter_episode_length.get_raw_data(), replay_buffer
