import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# DDQN
FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole/ddqn_vary_trained_on'
plot_name = 'cartpole_ddqn_vary_hp_both_lengths.eps'
title = "Trained synth. env. with DDQN"

# Transfer discrete TD3
FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole' \
           '/ddqn_to_td3_discrete_vary_td3HPs_variation_experiments/learned_temp_init_1_tanh_hard_True_lr_5e-4'
plot_name = 'ddqn_to_td3_discrete_gumbel_learned_temp_tanh_both_lengths.eps'
title = "Transfer DDQN -> TD3"

# Transfer discrete TD3 with 5 / 10 best models
FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole' \
           '/ddqn_to_td3_discrete_vary_td3HPs_variation_experiments/learned_temp_init_1_tanh_hard_True_lr_5e-4'
plot_name = 'ddqn_to_td3_discrete_gumbel_learned_temp_tanh_new.eps'
title = "Transfer DDQN -> TD3"

FILE_LIST = ['0.pt', '2.pt', '1.pt']
# FILE_LIST = ['0.pt', '1.pt']

ddqn_mean_train_steps = [16887.6925, 6818.57, 6379.5075]
ddqn_std_train_steps = [24925.0562208, 2339.505055, 3162.9542706]

dueling_ddqn_mean_train_steps = [12745.27, 6781.045, 6502.5125]
dueling_ddqn_std_train_steps = [14972.211664, 2198.149523570906, 3209.8083018]

#  ddqn_to_td3_discrete_vary_layer_norm_2_learned_temp
td3_mean_train_steps = [17874.925, 5832.0975, 5371.035]
td3_std_train_steps = [17834.68171216899, 1576.944465729136, 2414.505099140401]

# mode 2, 5 best models (see syn_env_evaluate_cartpole_vary_hp_2_TD3_discrete.py for which one), 4k evals (80(agents_num)*5(models)*10(
# evals per model))
td3_mean_train_steps[1] = 6287.5
td3_std_train_steps[1] = 1970.6455160682756
key = "2_5_best_filtered_models"
subtitle = "(5 best)"
FILE_LIST = ['0.pt', '2_5_best_filtered_models.pt', '1.pt']
plot_name = 'ddqn_to_td3_discrete_vary_layer_norm_2_filtered_models_best_5_4k_evals.eps'

# mode 2, 10 best models (see syn_env_evaluate_cartpole_vary_hp_2_TD3_discrete.py), 4k evals (agents_num 80)
# td3_mean_train_steps[1] = 5968.5776
# td3_std_train_steps[1] = 1814.0397236537435
# key = "2_10_best_filtered_models"
# subtitle = "(10 best)"
# FILE_LIST = ['0.pt', '2_10_best_filtered_models.pt', '1.pt']
# plot_name = 'ddqn_to_td3_discrete_vary_layer_norm_2_filtered_models_best_10_4k_evals.eps'

mean_train_steps = td3_mean_train_steps
std_train_steps = td3_std_train_steps

if __name__ == "__main__":
    data_list = []
    mean_list = []
    episode_num_needed_means = []
    episode_num_needed_stds = []
    for file in FILE_LIST:
        file_path = os.path.join(FILE_DIR, file)
        save_dict = torch.load(file_path)
        reward_list = save_dict['reward_list']

        if key in file:
            # keys got falsely named in the beginning of experiment:
            # train_steps were named num_episodes
            # new models are now correctly using keys --> mapping needed
            mean_episode_num = np.mean(save_dict["episode_length_needed"])
            std_episode_num = np.std(save_dict["episode_length_needed"])
        else:
            mean_episode_num = np.mean(save_dict["train_steps_needed"])
            std_episode_num = np.std(save_dict["train_steps_needed"])

        reward_list_single = []
        for r_list in reward_list:
            reward_list_single += r_list

        data_list.append(reward_list_single)
        if key in file:
            mean_list.append('mean {}: {:.2f}'.format(subtitle, (statistics.mean(reward_list_single))))
        else:
            mean_list.append('mean: {:.2f}'.format((statistics.mean(reward_list_single))))

        episode_num_needed_means.append(mean_episode_num)
        episode_num_needed_stds.append(std_episode_num)

    data_dict = {
        'train: real  / HP: varying\n(mean num episodes: {:.2f}$\pm${:.2f})\n(mean train steps: {:.2f}$\pm${:.2f})'.format(
            episode_num_needed_means[0], episode_num_needed_stds[0],
            mean_train_steps[0], std_train_steps[0]): data_list[0],

        'train: synth. / HP: varying\n({:.2f}$\pm${:.2f})\n({:.2f}$\pm${:.2f})'.format(
            episode_num_needed_means[1], episode_num_needed_stds[1],
            mean_train_steps[1], std_train_steps[1]): data_list[1],

        'train: synth. / HP: fixed\n({:.2f}$\pm${:.2f})\n({:.2f}$\pm${:.2f})'.format(
            episode_num_needed_means[2], episode_num_needed_stds[2],
            mean_train_steps[2], std_train_steps[2]): data_list[2]
    }

    data_dict = dict([(k, pd.Series(v)) for k, v in data_dict.items()])

    df = pd.DataFrame(data=data_dict)
    plt.figure(dpi=600, figsize=(7.5, 3))
    sns.set_context(rc={
        "font.size": 8.5,
        "axes.titlesize": 8,
        "axes.labelsize": 8
    })
    ax = sns.violinplot(data=df, cut=0, inner=None)
    plt.ylabel(title + '\ncumulative reward')

    for x, y, mean in zip([0, 1, 2], [220, 220, 220], mean_list):
        plt.text(x, y, mean, ha='center', va='center')

    plt.savefig(os.path.join(FILE_DIR, plot_name), bbox_inches='tight')
    plt.show()
