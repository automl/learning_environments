import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from utils import barplot_err

FILE_DIRS = []
FILE_LISTS = []
PLOT_NAMES = []
TITLES = []
MEAN_TRAIN_STEPS = []
STD_TRAIN_STEPS = []

FILE_DIRS.append('/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole/ddqn_vary_trained_on')
PLOT_NAMES.append('cartpole_ddqn_vary_hp_both_lengths.eps')
TITLES.append("Trained synth. env. with DDQN")
FILE_LISTS.append(['0.pt', '2.pt', '1.pt'])

FILE_DIRS.append('/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole/ddqn_to_duelingddqn_vary')
PLOT_NAMES.append('cartpole_ddqn_to_duelingddqn_vary_hp_both_lengths.eps')
TITLES.append("Transfer DDQN -> Dueling DDQN")
FILE_LISTS.append(['0.pt', '2.pt', '1.pt'])

FILE_DIRS.append('/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole' \
                 '/ddqn_to_td3_discrete_vary_td3HPs_variation_experiments/learned_temp_init_1_tanh_hard_True_lr_5e-4')
PLOT_NAMES.append('ddqn_to_td3_discrete_gumbel_learned_temp_tanh_both_lengths.eps')
TITLES.append("Transfer DDQN -> TD3")
FILE_LISTS.append(['0.pt', '2.pt', '1.pt'])

plot_name = "CP_vary_hp_merged_plots.pdf"

ddqn_mean_train_steps = [16887.6925, 6818.57, 6379.5075]
ddqn_std_train_steps = [24925.0562208, 2339.505055, 3162.9542706]

MEAN_TRAIN_STEPS.append(ddqn_mean_train_steps)
STD_TRAIN_STEPS.append(ddqn_std_train_steps)

dueling_ddqn_mean_train_steps = [12745.27, 6781.045, 6502.5125]
dueling_ddqn_std_train_steps = [14972.211664, 2198.149523570906, 3209.8083018]

MEAN_TRAIN_STEPS.append(dueling_ddqn_mean_train_steps)
STD_TRAIN_STEPS.append(dueling_ddqn_std_train_steps)

#  ddqn_to_td3_discrete_vary_layer_norm_2_learned_temp
td3_mean_train_steps = [17874.925, 5832.0975, 5371.035]
td3_std_train_steps = [17834.68171216899, 1576.944465729136, 2414.505099140401]

# mode 2, 5 best models (see syn_env_evaluate_cartpole_vary_hp_2_TD3_discrete.py for which one), 4k evals (80(agents_num)*5(models)*10(
# evals per model))
# td3_mean_train_steps[1] = 6287.5
# td3_std_train_steps[1] = 1970.6455160682756
# TITLES[2] = "Transfer DDQN -> TD3 (5 best)"
# FILE_LISTS[2] = ['0.pt', '2_5_best_filtered_models.pt', '1.pt']
# plot_name = 'CP_vary_hp_merged_plots_best_5_dtd3.pdf'

key = "2_5_best_filtered_models"  # don't comment this line
MEAN_TRAIN_STEPS.append(td3_mean_train_steps)
STD_TRAIN_STEPS.append(td3_std_train_steps)


if __name__ == "__main__":
    fig, axes = plt.subplots(figsize=(15, 5), ncols=3, nrows=2, sharex="row", sharey="row", gridspec_kw={
            'height_ratios': [2, 1.2]
            })

    for i, data in enumerate(zip(FILE_DIRS, FILE_LISTS, PLOT_NAMES, TITLES, MEAN_TRAIN_STEPS, STD_TRAIN_STEPS)):
        FILE_DIR, FILE_LIST, _, title, mean_train_steps, std_train_steps = data

        data_list = []
        mean_list = []
        episode_num_needed_means = []
        episode_num_needed_stds = []
        for j, file in enumerate(FILE_LIST):
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
            if i == 0 and j == 0:
                mean_list.append('mean reward: {:.2f}'.format((statistics.mean(reward_list_single))))
            else:
                mean_list.append('{:.2f}'.format((statistics.mean(reward_list_single))))

            episode_num_needed_means.append(mean_episode_num)
            episode_num_needed_stds.append(std_episode_num)

        data_dict = {
                'train: real, HP: varying': data_list[0],

                'train: synth., HP: varying': data_list[1],

                'train: synth., HP: fixed': data_list[2]
                }

        df = pd.DataFrame(data=data_dict)
        df = df.melt(value_name="cumulative rewards", var_name="type")

        sns.set_context(rc={
                "font.size": 11,
                "axes.titlesize": 11,
                "axes.labelsize": 11,
                "legend.fontsize": 11,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9
                })

        if i == 0:
            title += " (CartPole-v0)"

        g = sns.kdeplot(x="cumulative rewards", hue="type", data=df, ax=axes[0, i])
        g.set_title(title)

        if i == 0:
            # remove legend title with hue-kdeplot
            g.axes.get_legend().set_title("")
            axes[0, i].set_xlabel('cumulative rewards')
            axes[0, i].set_ylabel('Density')
        else:
            g.axes.get_legend().set_visible(False)
            axes[0, i].get_xaxis().get_label().set_visible(False)
            axes[0, i].get_yaxis().get_label().set_visible(False)

        barplot_df = pd.DataFrame({
                "type": ["steps", "episodes", "steps", "episodes", "steps", "episodes"],
                "method": ["train: real\nHP: varying", "train: real\nHP: varying", "train: synth.\nHP: varying",
                           "train: synth.\nHP: varying", "train: synth.\nHP: fixed", "train: synth.\nHP: fixed"],
                "means": [mean_train_steps[0], episode_num_needed_means[0], mean_train_steps[1], episode_num_needed_means[1],
                          mean_train_steps[2], episode_num_needed_means[2]],
                "std dev": [std_train_steps[0], episode_num_needed_stds[0], std_train_steps[1], episode_num_needed_stds[1],
                            std_train_steps[2], episode_num_needed_stds[2]]
                })

        scale = 100
        barplot_df['means'] = np.where(barplot_df['type'] == 'episodes', barplot_df['means'] * scale, barplot_df['means'])
        barplot_df['std dev'] = np.where(barplot_df['type'] == 'episodes', barplot_df['std dev'] * scale, barplot_df['std dev'])

        # clrs = ["blue", "lightblue", "orange", "lightorange", "green", "lightgreen"]

        p = barplot_err(x="method", y="means", yerr="std dev", hue="type", errwidth=1., capsize=.05, data=barplot_df, ax=axes[1, i],
                        palette=sns.color_palette("Paired"))

        axis_left = axes[1, i]
        axis_right = axis_left.twinx()
        axis_right.set_ylim(axis_left.get_ylim())
        axis_right.set_yticklabels(np.round(axis_left.get_yticks() / scale, 1).astype(int))

        if i == 0:
            p.axes.get_legend().set_title("")
            axis_left.set_ylabel("mean train steps")
            axis_right.set_ylabel("mean train episodes", rotation=-90, labelpad=12)
            axis_left.get_xaxis().get_label().set_visible(False)

        else:
            p.axes.get_xaxis().get_label().set_visible(False)
            p.axes.get_legend().set_visible(False)

            # remove tick labels from 2nd and 3rd plot but keep it in first while sharedx is active
            axes[1, i].tick_params(labelbottom=False)

            axis_left.get_yaxis().get_label().set_visible(False)
            axis_left.get_xaxis().get_label().set_visible(False)
            axis_right.get_yaxis().get_label().set_visible(False)
            axis_right.get_yaxis().set_ticklabels([])
            axes[1, i].get_xaxis().get_label().set_visible(False)

        for x, y, mean in zip([0, 1, 2], [220, 220, 220], mean_list):
            plt.text(x, y, mean, ha='center', va='top', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "transfer_experiments/cartpole", plot_name), bbox_inches='tight')
    plt.show()
