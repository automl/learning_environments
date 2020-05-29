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
        self.reward = torch.zeros((max_size,1))
        self.done = torch.zeros((max_size,1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward.squeeze()
        self.done[self.ptr] = done.squeeze()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # for TD3
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[ind].to(device).detach(),
            self.action[ind].to(device).detach(),
            self.next_state[ind].to(device).detach(),
            self.reward[ind].to(device).detach(),
            self.done[ind].to(device).detach()
        )

    # for PPO
    def get_all(self):
        return (
            self.state[:self.size].to(device).detach(),
            self.action[:self.size].to(device).detach(),
            self.next_state[:self.size].to(device).detach(),
            self.reward[:self.size].to(device).detach(),
            self.done[:self.size].to(device).detach()
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
            print(self.print_str + str(np.mean(self.vals[:self.size])) + '   Total updates: ' + str(self.it))
