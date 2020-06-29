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

        self.last_state = torch.zeros((max_size, state_dim))
        self.last_action = torch.zeros((max_size, action_dim))
        self.state = torch.zeros((max_size, state_dim))
        self.action = torch.zeros((max_size, action_dim))
        self.next_state = torch.zeros((max_size, state_dim))
        self.reward = torch.zeros((max_size, 1))
        self.done = torch.zeros((max_size, 1))
        self.input_seed = torch.zeros((max_size, 1))

    def add(self, last_state, last_action, state, action, next_state, reward, done, input_seed):
        self.last_state[self.ptr] = last_state.detach()
        self.last_action[self.ptr] = last_action.detach()
        self.state[self.ptr] = state.detach().detach()
        self.action[self.ptr] = action.detach()
        self.next_state[self.ptr] = next_state.detach()
        self.reward[self.ptr] = reward.squeeze().detach()
        self.done[self.ptr] = done.squeeze().detach()
        self.input_seed[self.ptr] = input_seed.squeeze().detach()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # for TD3
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.last_state[ind].to(device).detach(),
            self.last_action[ind].to(device).detach(),
            self.state[ind].to(device).detach(),
            self.action[ind].to(device).detach(),
            self.next_state[ind].to(device).detach(),
            self.reward[ind].to(device).detach(),
            self.done[ind].to(device).detach(),
            self.input_seed[ind].to(device).detach()
        )

    # for PPO
    def get_all(self):
        return (
            self.last_state[: self.size].to(device).detach(),
            self.last_action[: self.size].to(device).detach(),
            self.state[: self.size].to(device).detach(),
            self.action[: self.size].to(device).detach(),
            self.next_state[: self.size].to(device).detach(),
            self.reward[: self.size].to(device).detach(),
            self.done[: self.size].to(device).detach(),
            self.input_seed[: self.size].to(device).detach()
        )

    # for PPO
    def clear(self):
        self.__init__(self.state_dim, self.action_dim, self.max_size)


class AverageMeter:
    def __init__(self, print_str):
        self.print_str = print_str
        self.vals = []
        self.ptr = 0
        self.size = 0
        self.it = 0

    def update(self, val, print_rate=10):
        self.vals.append(val)
        self.it += 1

        if self.it % print_rate == 0:
            mean_val = self._mean(print_rate)
            out = self.print_str + "{:15.4f} {:>20} {}".format(mean_val, "Total updates: ", self.it)
            print(out)

    def get_mean(self, num=10):
        return self._mean(num)

    def get_raw_data(self):
        return self.vals

    def _mean(self, num):
        vals = self.vals[max(len(self.vals)-num-1, 0) : len(self.vals)]
        return sum(vals) / len(vals)



def print_abs_param_sum(model, model_name=''):
    sm = 0
    for param in model.parameters():
        sm += abs(param).sum()
    print(model_name + ' ' + str(sm))
