import os

import matplotlib.pyplot as plt
import numpy as np
import torch

LOG_FILES = ['../results/cliff_compare_reward_envs/best1.pt',
             '../results/cliff_compare_reward_envs/best2.pt',
             '../results/cliff_compare_reward_envs/best5.pt',
             '../results/cliff_compare_reward_envs/best6.pt',
             '../results/cliff_compare_reward_envs/best0.pt',
             # '../results/cliff_compare_reward_envs/best-1.pt',
             '../results/cliff_compare_reward_envs/best-1_opt.pt'
             ]

LEGEND = [
        'QL + exc. pot. RN',
        'QL + add. pot. RN',
        'QL + exc. non-pot. RN',
        'QL + add. non-pot. RN',
        'QL',
        'QL + count-based expl.',
        # 'QL + count-based expl. (tuned)'
        ]

STD_MULT = 0.2
MIN_STEPS = 5000


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
            print(sum(episode_lengths))
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

    for mean, std in proc_data:
        plt.plot(mean, linewidth=1)

    for mean, std in proc_data:
        plt.fill_between(x=range(len(mean)), y1=mean - std * STD_MULT, y2=mean + std * STD_MULT, alpha=0.1)

    plt.legend(LEGEND, fontsize=7)
    # plt.xlim(0,99)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.title('Cliff Walking')
    plt.xlabel('steps')
    plt.xlim(0, MIN_STEPS)
    plt.ylim(-100, -10)
    plt.ylabel('cumulative reward')
    base_dir = os.path.dirname(LOG_FILES[0])
    plt.savefig(os.path.join(base_dir, savefig_name))
    # plt.show()


if __name__ == "__main__":
    proc_data = get_data()
    plot_data(proc_data=proc_data, savefig_name=f'gridworld_compare_reward_env.pdf')
    plot_data(proc_data=proc_data, savefig_name=f'gridworld_compare_reward_env.png')
