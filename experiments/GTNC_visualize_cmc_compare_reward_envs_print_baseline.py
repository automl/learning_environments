import matplotlib.pyplot as plt
import torch
import numpy as np

LOG_FILE = '../results/cmc_compare_reward_envs/best0.pt'


def get_data():
    data = torch.load(LOG_FILE)
    reward_list = data['reward_list']

    print(len(reward_list))
    count = 0
    for rewards in reward_list:
        if max(rewards) > 90:
            count += 1

    print(count)


if __name__ == "__main__":
    proc_data = get_data()
