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

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1), dtype=np.bool)

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state.data.numpy()
        self.action[self.ptr] = action.data.numpy()
        self.next_state[self.ptr] = next_state.data.numpy()
        self.reward[self.ptr] = reward.squeeze().data.numpy()
        self.done[self.ptr] = done.squeeze().data.numpy()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # for TD3
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.from_numpy(self.state[ind]).float().to(device),
            torch.from_numpy(self.action[ind]).float().to(device),
            torch.from_numpy(self.next_state[ind]).float().to(device),
            torch.from_numpy(self.reward[ind]).float().to(device),
            torch.from_numpy(self.done[ind]).bool().to(device),
        )

    # for PPO
    def get_all(self):
        return (
            torch.from_numpy(self.state[: self.size]).float().to(device),
            torch.from_numpy(self.action[: self.size]).float().to(device),
            torch.from_numpy(self.next_state[: self.size]).float().to(device),
            torch.from_numpy(self.reward[: self.size]).float().to(device),
            torch.from_numpy(self.done[: self.size]).float().to(device),
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
