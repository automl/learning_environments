import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
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
        self.state[self.ptr] = state.detach().detach()
        self.action[self.ptr] = action.detach()
        self.next_state[self.ptr] = next_state.detach()
        self.reward[self.ptr] = reward.squeeze().detach()
        self.done[self.ptr] = done.squeeze().detach()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # for TD3 / gym
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[ind].to(device).detach(),
            self.action[ind].to(device).detach(),
            self.next_state[ind].to(device).detach(),
            self.reward[ind].to(device).detach(),
            self.done[ind].to(device).detach(),
        )

    # for PPO
    def get_all(self):
        return (
            self.state[: self.size].to(device).detach(),
            self.action[: self.size].to(device).detach(),
            self.next_state[: self.size].to(device).detach(),
            self.reward[: self.size].to(device).detach(),
            self.done[: self.size].to(device).detach(),
        )

    def get_size(self):
        return self.size

    # for PPO
    def clear(self):
        self.__init__(self.state_dim, self.action_dim, self.max_size)  #


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
            mean_val = self._mean(print_rate)
            out = self.print_str + "{:15.6f} {:>25} {}".format(mean_val, "Total updates: ", self.it)
            print(out)

    def get_mean(self, num=10):
        return self._mean(num)

    def get_raw_data(self):
        return self.vals

    def _mean(self, num):
        vals = self.vals[max(len(self.vals) - num, 0) : len(self.vals)]
        return sum(vals) / (len(vals) + 1e-9)


def print_abs_param_sum(model, name=""):
    sm = 0
    for param in model.parameters():
        sm += abs(param).sum()
    print(name + " " + str(sm))


def print_avg_pairwise_dist(vec_list, name=""):
    n = len(vec_list)
    if n < 2:
        return 0

    count = 0
    avg_dist = 0
    for i in range(n):
        for k in range(i, n):
            count += 1
            avg_dist += torch.dist(vec_list[i], vec_list[k])

    print(name + " " + str(avg_dist/count))





