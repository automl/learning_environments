import os

import matplotlib.pyplot as plt
import numpy as np
import torch

LOG_FILES = [
        '../results/cmc_compare_reward_envs/best_transfer_algo1.pt',  # running
        '../results/cmc_compare_reward_envs/best_transfer_algo2.pt',
        '../results/cmc_compare_reward_envs/best_transfer_algo5.pt',
        '../results/cmc_compare_reward_envs/best_transfer_algo6.pt',
        '../results/cmc_compare_reward_envs/best_transfer_algo0.pt',
        '../results/cmc_compare_reward_envs/best_transfer_algo-1.pt'
        ]

LEGEND = [
        'PPO + exc. pot. RN',
        'PPO + add. pot. RN',
        'PPO + exc. non-pot. RN',
        'PPO + add. non-pot. RN',
        'PPO',
        'PPO + ICM',
        ]

STD_MULT = .2
# STD_MULT = 1.
MIN_STEPS = 200000

def get_data():
    list_data = []
    for log_file in LOG_FILES:
        data = torch.load(log_file)
        list_data.append((data['reward_list'], data['episode_length_list']))
        model_num = data['model_num']
        model_agents = data['model_agents']

    sums_eps_len = []
    sums_eps_len_per_model = []
    min_steps = float('Inf')
    # get minimum number of evaluations
    for reward_list, episode_length_list in list_data:
        sums_eps_len_per_model_i = []
        for episode_lengths in episode_length_list:
            print(len(episode_lengths))
            sums_eps_len.append(sum(episode_lengths))
            sums_eps_len_per_model_i.append(sum(episode_lengths))
            min_steps = min(min_steps, sum(episode_lengths))
        sums_eps_len_per_model.append(sums_eps_len_per_model_i)

    print("# of episode lens > MIN_STEPS: ", sum(np.asarray(sums_eps_len) >= MIN_STEPS))
    print("# of episode lens < MIN_STEPS: ", sum(np.asarray(sums_eps_len) < MIN_STEPS))
    min_steps = max(min_steps, MIN_STEPS)
    # convert data from episodes to steps
    proc_data = []

    for reward_list, episode_length_list in list_data:
        np_data = np.zeros([model_num * model_agents, min_steps])

        for it, data in enumerate(zip(reward_list, episode_length_list)):
            rewards, episode_lengths = data

            concat_list = []
            rewards = rewards

            for i in range(len(episode_lengths)):
                concat_list += [rewards[i]] * episode_lengths[i]

            while len(concat_list) < min_steps:
                concat_list.append(concat_list[-1])

            np_data[it] = np.array(concat_list[:min_steps])

        mean = np.mean(np_data, axis=0)
        std = np.std(np_data, axis=0)

        proc_data.append((mean, std))

    return proc_data


def plot_data(proc_data, savefig_name):
    fig, ax = plt.subplots(dpi=600, figsize=(5, 4))
    colors = []
    #
    # for mean, _ in data_w:
    #     plt.plot(mean_w)
    #     colors.append(plt.gca().lines[-1].get_color())

    for mean, std in proc_data:
        plt.plot(mean)

    for mean, std in proc_data:
        plt.fill_between(x=range(len(mean)), y1=mean - std * STD_MULT, y2=mean + std * STD_MULT, alpha=0.1)

    plt.legend(LEGEND, fontsize=7)
    # plt.xlim(0,99)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.title('MountainCarContinuous-v0 Transfer')
    plt.xlabel('steps')
    plt.xlim(0, MIN_STEPS)
    # freq_x_ticks = np.arange(0, MIN_STEPS, 33000)
    # plt.xticks(freq_x_ticks)
    # ax.xaxis.set_tick_params(labelsize='small')
    plt.ylim(-75, 100)
    plt.ylabel('cumulative reward')
    base_dir = os.path.dirname(LOG_FILES[0])
    plt.savefig(os.path.join(base_dir, savefig_name))
    plt.show()


if __name__ == "__main__":
    proc_data = get_data()
    plot_data(proc_data=proc_data, savefig_name=f'cmc_transfer_algo_{MIN_STEPS}_steps.png')
