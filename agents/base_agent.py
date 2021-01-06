import time
import torch
import torch.nn as nn
import statistics
from utils import AverageMeter, ReplayBuffer


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
        self.early_out_episode = agent_config["early_out_episode"]
        self.early_out_virtual_diff = agent_config["early_out_virtual_diff"]

        self.render_env = config["render_env"]
        self.device = config["device"]


    def time_is_up(self, avg_meter_reward, max_episodes, time_elapsed, time_remaining):
        if time_elapsed > time_remaining:
            print("timeout")
            # fill remaining rewards with minimum reward achieved so far
            if len(avg_meter_reward.get_raw_data()) == 0:
                avg_meter_reward.update(-1e9)
            while len(avg_meter_reward.get_raw_data()) < max_episodes:
                avg_meter_reward.update(min(avg_meter_reward.get_raw_data()), print_rate=1e9)
            return True
        else:
            return False


    def env_solved(self, env, avg_meter_reward, episode, real_env=None):
        avg_reward = avg_meter_reward.get_mean(num=self.early_out_num)
        avg_reward_last = avg_meter_reward.get_mean_last(num=self.early_out_num)
        if env.is_virtual_env():
            if abs(avg_reward - avg_reward_last) / abs(avg_reward_last + 1e-9) < self.early_out_virtual_diff and \
                    episode > self.init_episodes + self.early_out_num:
                #print("early out on virtual env after {} episodes with an average reward of {}".format(episode + 1, avg_reward))
                return True
        elif real_env is not None:
            avg_reward_test, _ = self.test(real_env)
            avg_reward = statistics.mean(avg_reward_test)
            if avg_reward >= env.get_solved_reward():
                return True
        else:
            if avg_reward >= env.get_solved_reward():
                #print("early out on real env after {} episodes with an average reward of {}".format(episode + 1, avg_reward))
                return True

        return False


    def train(self, env, real_env=None, time_remaining=1e9):
        time_start = time.time()

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

        replay_buffer = ReplayBuffer(state_dim=sd, action_dim=ad, device=self.device, max_size=self.rb_size)

        avg_meter_reward = AverageMeter(print_str="Average reward: ")

        env.set_agent_params(same_action_num=self.same_action_num, gamma=self.gamma)

        # training loop
        for episode in range(self.train_episodes):
            # early out if timeout
            if self.time_is_up(avg_meter_reward=avg_meter_reward,
                               max_episodes=self.train_episodes,
                               time_elapsed=time.time() - time_start,
                               time_remaining=time_remaining):
                break

            if hasattr(self, 'update_parameters_per_episode'):
                self.update_parameters_per_episode(episode=episode)

            state = env.reset()
            episode_reward = 0

            for t in range(0, env.max_episode_steps(), self.same_action_num):
                action = self.select_train_action(state=state, env=env, episode=episode)

                # live view
                if self.render_env and episode % 10 == 0:
                    env.render()

                # state-action transition
                # required due to gumble softmax in td3 discrete
                if discretize_action:
                    next_state, reward, done = env.step(action=action.argmax().unsqueeze(0))
                else:
                    next_state, reward, done = env.step(action=action)
                replay_buffer.add(state=state, action=action, next_state=next_state, reward=reward, done=done)
                state = next_state
                episode_reward += reward

                # train
                if episode > self.init_episodes:
                    self.learn(replay_buffer=replay_buffer, env=env, episode=episode)

                if done > 0.5:
                    break

            # logging
            avg_meter_reward.update(episode_reward, print_rate=self.print_rate)

            # quit training if environment is solved
            if episode > self.init_episodes and episode % self.early_out_episode == 0 and self.env_solved(env=env, avg_meter_reward=avg_meter_reward, episode=episode, real_env=real_env):
                break

        env.close()

        return avg_meter_reward.get_raw_data(), replay_buffer


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

            # training loop
            for episode in range(self.test_episodes):
                # early out if timeout
                if self.time_is_up(avg_meter_reward=avg_meter_reward,
                                   max_episodes=self.test_episodes,
                                   time_elapsed=time.time() - time_start,
                                   time_remaining=time_remaining):
                    break

                state = env.reset()
                episode_reward = 0

                for t in range(0, env.max_episode_steps(), self.same_action_num):
                    action = self.select_test_action(state, env)

                    # live view
                    if self.render_env and episode % 10 == 0:
                        env.render()

                    # state-action transition
                    # required due to gumble softmax in td3 discrete
                    if discretize_action:
                        next_state, reward, done = env.step(action=action.argmax().unsqueeze(0))
                    else:
                        next_state, reward, done = env.step(action=action)
                    replay_buffer.add(state=state, action=action, next_state=next_state, reward=reward, done=done)
                    state = next_state
                    episode_reward += reward

                    if done > 0.5:
                        break

                # logging
                avg_meter_reward.update(episode_reward.item(), print_rate=self.print_rate)

            env.close()

        return avg_meter_reward.get_raw_data(), replay_buffer
