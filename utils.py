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

    def add(self, last_state, last_action, state, action, next_state, reward, done):
        self.last_state[self.ptr] = last_state.detach()
        self.last_action[self.ptr] = last_action.detach()
        self.state[self.ptr] = state.detach().detach()
        self.action[self.ptr] = action.detach()
        self.next_state[self.ptr] = next_state.detach()
        self.reward[self.ptr] = reward.squeeze().detach()
        self.done[self.ptr] = done.squeeze().detach()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # for TD3
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.last_state[ind].to(device),
            self.last_action[ind].to(device),
            self.state[ind].to(device),
            self.action[ind].to(device),
            self.next_state[ind].to(device),
            self.reward[ind].to(device),
            self.done[ind].to(device)
        )

    # for PPO
    def get_all(self):
        return (
            self.last_state[: self.size].to(device),
            self.last_action[: self.size].to(device),
            self.state[: self.size].to(device),
            self.action[: self.size].to(device),
            self.next_state[: self.size].to(device),
            self.reward[: self.size].to(device),
            self.done[: self.size].to(device)
        )

    # for PPO
    def clear(self):
        self.__init__(self.state_dim, self.action_dim, self.max_size)


class AverageMeter:
    def __init__(self, buffer_size, update_rate, print_str):
        self.buffer_size = buffer_size
        self.update_rate = update_rate
        self.print_str = print_str
        self.vals = np.zeros(buffer_size)
        self.ptr = 0
        self.size = 0
        self.it = 1

    def update(self, val):
        self.vals[self.ptr] = val
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        self.it += 1

        if self.it % self.update_rate == 0:
            out = self.print_str + "{:15.8f} {:>25} {}".format(
                np.mean(self.vals[: self.size]), "Total updates: ", self.it
            )
            print(out)
