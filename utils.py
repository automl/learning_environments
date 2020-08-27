import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.device = device
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0

        self.state = torch.zeros((max_size, state_dim))
        self.action = torch.zeros((max_size, action_dim))
        self.next_state = torch.zeros((max_size, state_dim))
        self.reward = torch.zeros((max_size, 1))
        self.done = torch.zeros((max_size, 1))

    # for TD3 / PPO / gym
    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state.detach()
        self.action[self.ptr] = action.detach()
        self.next_state[self.ptr] = next_state.detach()
        self.reward[self.ptr] = reward.squeeze().detach()
        self.done[self.ptr] = done.squeeze().detach()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # for TD3 / gym
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self._sample_idx(idx)

    def _sample_idx(self, idx):
        return (
            self.state[idx].to(self.device).detach(),
            self.action[idx].to(self.device).detach(),
            self.next_state[idx].to(self.device).detach(),
            self.reward[idx].to(self.device).detach(),
            self.done[idx].to(self.device).detach(),
        )

    # for PPO
    def get_all(self):
        return (
            self.state[:self.size].to(self.device).detach(),
            self.action[:self.size].to(self.device).detach(),
            self.next_state[:self.size].to(self.device).detach(),
            self.reward[:self.size].to(self.device).detach(),
            self.done[:self.size].to(self.device).detach(),
        )

    def merge(self, other_replay_buffer):
        states, actions, next_states, rewards, dones = other_replay_buffer.get_all()
        for i in range(len(states)):
            self.add(states[i], actions[i], next_states[i], rewards[i], dones[i])

    def get_size(self):
        return self.size

    # for PPO
    def clear(self):
        self.__init__(state_dim=self.state_dim, action_dim=self.action_dim, device=self.device, max_size=self.max_size)


class AverageMeter:
    def __init__(self, print_str):
        self.print_str = print_str
        self.vals = []
        self.it = 0

    def update(self, val, print_rate=10):
        # formatting from torch/numpy to float
        if torch.is_tensor(val):
            val = val.cpu().data.numpy()
            if not np.ndim(val) == 0:
                val = val[0]

        self.vals.append(val)
        self.it += 1

        if self.it % print_rate == 0:
            mean_val = self._mean(num=print_rate, ignore_last=0)
            out = self.print_str + "{:15.6f} {:>25} {}".format(mean_val, "Total updates: ", self.it)
            print(out)

    def get_mean(self, num=10):
        return self._mean(num, ignore_last=0)

    def get_mean_last(self, num=10):
        return self._mean(num, ignore_last=num)

    def get_raw_data(self):
        return self.vals

    def _mean(self, num, ignore_last):
        vals = self.vals[max(len(self.vals) - num - ignore_last, 0) : max(len(self.vals) - ignore_last, 0)]
        return sum(vals) / (len(vals) + 1e-9)


def time_is_up(avg_meter_reward, max_episodes, time_elapsed, time_remaining):
    if time_elapsed > time_remaining:
        print("timeout")
        # fill remaining rewards with minimum reward achieved so far
        if len(avg_meter_reward.get_raw_data()) == 0:
            avg_meter_reward.update(0)

        while len(avg_meter_reward.get_raw_data()) < max_episodes:
            avg_meter_reward.update(min(avg_meter_reward.get_raw_data()), print_rate=1e9)
        return True
    else:
        return False


def env_solved(agent, env, avg_meter_reward, episode):
    avg_reward = avg_meter_reward.get_mean(num=agent.early_out_num)
    avg_reward_last = avg_meter_reward.get_mean_last(num=agent.early_out_num)
    if env.is_virtual_env():
        if abs(avg_reward-avg_reward_last) / (avg_reward_last+1e-9) < agent.early_out_virtual_diff and episode > agent.init_episodes:
            print("early out after {} episodes with an average reward of {}".format(episode+1, avg_reward))
            return True
    else:
        if avg_reward >= env.env.solved_reward and episode > agent.init_episodes:
            print("early out after {} episodes with an average reward of {}".format(episode+1, avg_reward))
            return True

    return False


def calc_abs_param_sum(model):
    sm = 0
    for param in model.parameters():
        sm += abs(param).sum()
    return sm


def print_abs_param_sum(model, name=""):
    print(name + " " + str(calc_abs_param_sum(model)))






