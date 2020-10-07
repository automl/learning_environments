import time
import torch
import torch.nn as nn
from utils import AverageMeter, ReplayBuffer


class BaseAgent(nn.Module):
    def __init__(self, agent_name, state_dim, action_dim, config):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        agent_config = config["agents"][agent_name]
        self.train_episodes = agent_config["train_episodes"]
        self.test_episodes = agent_config["test_episodes"]
        self.init_episodes = agent_config["init_episodes"]
        self.same_action_num = agent_config["same_action_num"]
        self.print_rate = agent_config["print_rate"]
        self.early_out_num = agent_config["early_out_num"]
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


    def env_solved(self, env, avg_meter_reward, episode):
        avg_reward = avg_meter_reward.get_mean(num=self.early_out_num)
        avg_reward_last = avg_meter_reward.get_mean_last(num=self.early_out_num)
        if env.is_virtual_env():
            if abs(avg_reward - avg_reward_last) / abs(avg_reward_last + 1e-9) < self.early_out_virtual_diff and \
                    episode > self.init_episodes + self.early_out_num:
                #print("early out on virtual env after {} episodes with an average reward of {}".format(episode + 1, avg_reward))
                return True
        else:
            if avg_reward >= env.get_solved_reward() and episode > self.init_episodes:
                #print("early out on real env after {} episodes with an average reward of {}".format(episode + 1, avg_reward))
                return True

        return False


    def test(self, env, time_remaining=1e9, gtn_iteration=0):
        #self.plot_q_function(env)
        sd = 1 if env.has_discrete_state_space() else self.state_dim
        ad = 1 if env.has_discrete_action_space() else self.action_dim
        replay_buffer = ReplayBuffer(state_dim=sd, action_dim=ad, device=self.device, max_size=int(1e6))

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
                    action = self.select_test_action(state, env, gtn_iteration)

                    # live view
                    if self.render_env and episode % 10 == 0:
                        env.render()

                    # state-action transition
                    next_state, reward, done = env.step(action=action, same_action_num=self.same_action_num)
                    replay_buffer.add(state=state, action=action, next_state=next_state, reward=reward, done=done)
                    state = next_state
                    episode_reward += reward

                    if done > 0.5:
                        break

                # logging
                avg_meter_reward.update(episode_reward, print_rate=self.print_rate)

                # if self.env_solved(env=env, avg_meter_reward=avg_meter_reward, episode=episode):
                #     #print(path)
                #     #self.plot_q_function(env)
                #     break

            env.close()

        #self.plot_q_function(env)

        return avg_meter_reward.get_raw_data(), replay_buffer
