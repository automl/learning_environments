import time
import torch
import torch.nn as nn
from utils import ReplayBuffer, AverageMeter, time_is_up, env_solved


class BaseAgent(nn.Module):
    def __init__(self):
        super().__init__()

    def _train(self, env, time_remaining, update_parameters_per_episode_f, select_train_action_f, learn_f):
        time_start = time.time()

        if env.has_discrete_action_space():
            replay_buffer = ReplayBuffer(state_dim=env.get_state_dim(), action_dim=1, device=self.device, max_size=self.rb_size)
        else:
            replay_buffer = ReplayBuffer(state_dim=self.state_dim, action_dim=self.action_dim, device=self.device, max_size=self.rb_size)

        avg_meter_reward = AverageMeter(print_str="Average reward: ")

        # training loop
        for episode in range(self.train_episodes):
            # early out if timeout
            if time_is_up(avg_meter_reward=avg_meter_reward,
                          max_episodes=self.train_episodes,
                          time_elapsed=time.time() - time_start,
                          time_remaining=time_remaining):
                break

            update_parameters_per_episode_f(episode=episode)

            state = env.reset()
            episode_reward = 0

            for t in range(0, env.max_episode_steps(), self.same_action_num):
                action = select_train_action_f(state=state, env=env, episode=episode)

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
                    learn_f(replay_buffer)

                if done > 0.5:
                    break

            # logging
            avg_meter_reward.update(episode_reward, print_rate=self.print_rate)

            # quit training if environment is solved
            if env_solved(agent=self, env=env, avg_meter_reward=avg_meter_reward, episode=episode):
                break

        env.close()

        return avg_meter_reward.get_raw_data()


    def _test(self, env, select_test_action_f, time_remaining=1e9):
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
                    action = select_test_action_f(state)

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


    def _update_parameters_per_episode(self, episode):
        pass
