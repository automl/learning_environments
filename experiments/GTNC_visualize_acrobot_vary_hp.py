import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/acrobot/ddqn_to_duelingddqn_vary'
plot_name = 'acrobot_ddqn_to_duelingddqn_vary_hp_new.eps'
title = "Transfer DDQN -> Dueling DDQN"

# FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/acrobot/ddqn_vary_trained_on'
# plot_name = 'acrobot_ddqn_vary_hp.eps'
# title = "Trained synth. env. with DDQN"

# FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/acrobot/ddqn_to_td3_discrete_vary' \
#            '/learned_temp_relu_hard_false_config'
# plot_name = 'acrobot_ddqn_to_td3_vary_hp_learned_temp_relu_hard_false.eps'
# title = "Transfer DDQN -> TD3"

# FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/acrobot/correct_episode_length/ddqn_vary_trained_on'
# FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/acrobot/correct_episode_length/ddqn_to_duelingddqn_vary'
# FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/acrobot/correct_episode_length/ddqn_to_duelingddqn_vary/new'
# FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments'

# FILE_LIST = ['2.pt']
FILE_LIST = ['0.pt', '2.pt', '1.pt']
# FILE_LIST = ['0.pt', '1.pt']

if __name__ == "__main__":
    data_list = []
    mean_list = []
    episode_num_needed_means = []
    episode_num_needed_stds = []
    for file in FILE_LIST:
        file_path = os.path.join(FILE_DIR, file)
        save_dict = torch.load(file_path)
        reward_list = save_dict['reward_list']

        mean_episode_num = np.mean(save_dict["train_steps_needed"])
        std_episode_num = np.std(save_dict["train_steps_needed"])

        reward_list_single = []
        for r_list in reward_list:
            reward_list_single += r_list

        data_list.append(reward_list_single)
        mean_list.append('mean: {:.2f}'.format((statistics.mean(reward_list_single))))

        episode_num_needed_means.append(mean_episode_num)
        episode_num_needed_stds.append(std_episode_num)

    data_dict = {
        'train: real  / HP: varying\n(mean num episodes: {:.2f}$\pm${:.2f})'.format(
            episode_num_needed_means[0], episode_num_needed_stds[0]): data_list[0],
        'train: synth. / HP: varying\n({:.2f}$\pm${:.2f})'.format(
            episode_num_needed_means[1], episode_num_needed_stds[1]): data_list[1],
        'train: synth. / HP: fixed\n({:.2f}$\pm${:.2f})'.format(
            episode_num_needed_means[2], episode_num_needed_stds[2]): data_list[2]
    }

    df = pd.DataFrame(data=data_dict)
    plt.figure(dpi=600, figsize=(7.5, 3))
    sns.set_context(rc={
        "font.size": 8.5,
        "axes.titlesize": 8,
        "axes.labelsize": 8
    })
    ax = sns.violinplot(data=df, cut=0, inner=None)
    # ax = sns.violinplot(data=df, cut=0, inner=None).set_title(title, y=1.05)
    plt.ylabel(title + '\ncumulative reward')
    # plt.ylim(-500, -100)

    for x, y, mean in zip([0, 1, 2], [-15, -15, -15], mean_list):  # change to [-80, -80, -80] ?
        plt.text(x, y, mean, ha='center', va='center')

    plt.savefig(os.path.join(FILE_DIR, plot_name), bbox_inches='tight')
    plt.show()
