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

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state.detach()
        self.action[self.ptr] = action.detach()
        self.next_state[self.ptr] = next_state.detach()
        self.reward[self.ptr] = reward.squeeze().detach()
        self.done[self.ptr] = done.squeeze().detach()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

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

    def get_all(self):
        return (
            self.state[:self.size].to(self.device).detach(),
            self.action[:self.size].to(self.device).detach(),
            self.next_state[:self.size].to(self.device).detach(),
            self.reward[:self.size].to(self.device).detach(),
            self.done[:self.size].to(self.device).detach(),
        )

    def merge_buffer(self, other_replay_buffer):
        states, actions, next_states, rewards, dones = other_replay_buffer.get_all()
        self.merge_vectors(states=states, actions=actions, next_states=next_states, rewards=rewards, dones=dones)

    def merge_vectors(self, states, actions, next_states, rewards, dones):
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


def to_one_hot_encoding(normal, one_hot_dim):
    if torch.is_tensor(normal):
        normal = normal.squeeze()

    if not torch.is_tensor(normal):
        one_hot = torch.zeros(one_hot_dim)
        one_hot[int(normal)] = 1
    elif normal.dim() == 0 or (normal.dim() == 1 and len(normal) == 1):     # single torch value
        one_hot = torch.zeros(one_hot_dim)
        one_hot[int(normal.item())] = 1
    elif normal.dim() == 1:     # vector of values
        n = len(normal)
        one_hot = torch.zeros(n, one_hot_dim)
        for i in range(n):
            one_hot[i] = to_one_hot_encoding(normal[i], one_hot_dim)
    else:
        raise NotImplementedError('One hot encoding supported only for scalar values and 1D vectors')

    return one_hot


def from_one_hot_encoding(one_hot):
    return torch.tensor([torch.argmax(one_hot)])


def calc_abs_param_sum(model):
    sm = 0
    for param in model.parameters():
        sm += abs(param).sum()
    return sm


def print_abs_param_sum(model, name=""):
    print(name + " " + str(calc_abs_param_sum(model)))



