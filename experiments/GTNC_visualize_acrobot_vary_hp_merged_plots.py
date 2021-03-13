import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from utils import barplot_err

FILE_DIRS = []
PLOT_NAMES = []
TITLES = []
MEAN_TRAIN_STEPS = []
STD_TRAIN_STEPS = []

FILE_DIRS.append('/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/acrobot/ddqn_vary_trained_on')
PLOT_NAMES.append('acrobot_ddqn_vary_hp.eps')
TITLES.append("Trained synth. env. with DDQN")

FILE_DIRS.append('/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/acrobot/ddqn_to_duelingddqn_vary')
PLOT_NAMES.append('acrobot_ddqn_to_duelingddqn_vary_hp_both_lengths.eps')
TITLES.append("Transfer DDQN -> Dueling DDQN")

FILE_DIRS.append('/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/acrobot/ddqn_to_td3_discrete_vary' \
                 '/learned_temp_relu_hard_false_config')
PLOT_NAMES.append('acrobot_ddqn_to_td3_vary_hp_learned_temp_relu_hard_false.eps')
TITLES.append("Transfer DDQN -> TD3")

FILE_LIST = ['0.pt', '2.pt', '1.pt']

ddqn_mean_train_steps = [15388.76, 18240.325, 18177.8325]
ddqn_std_train_steps = [12820.303525556637, 6738.70980896, 8802.080576457]

MEAN_TRAIN_STEPS.append(ddqn_mean_train_steps)
STD_TRAIN_STEPS.append(ddqn_std_train_steps)

dueling_ddqn_mean_train_steps = [29531.1275, 18376.2125, 18540.66]
dueling_ddqn_std_train_steps = [79218.56009538576, 6401.7216619706105, 8945.47788686552]

MEAN_TRAIN_STEPS.append(dueling_ddqn_mean_train_steps)
STD_TRAIN_STEPS.append(dueling_ddqn_std_train_steps)

# MEAN_TRAIN_STEPS.append([50000, 50000, 50000])
# STD_TRAIN_STEPS.append([50000, 50000, 50000])

if __name__ == "__main__":

    fig, axes = plt.subplots(figsize=(10, 5), ncols=2, nrows=2, sharex="row", sharey="row", gridspec_kw={
            'height_ratios': [2, 1.2]
            })

    for i, data in enumerate(zip(FILE_DIRS, PLOT_NAMES, TITLES, MEAN_TRAIN_STEPS, STD_TRAIN_STEPS)):
        FILE_DIR, plot_name, title, mean_train_steps, std_train_steps = data

        data_list = []
        mean_list = []
        episode_num_needed_means = []
        episode_num_needed_stds = []
        for j, file in enumerate(FILE_LIST):
            file_path = os.path.join(FILE_DIR, file)
            save_dict = torch.load(file_path)
            reward_list = save_dict['reward_list']

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
            title += " (Acrobot-v1)"

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
                "type": [
                        "steps", "episodes",
                        "steps", "episodes",
                        "steps", "episodes"
                        ],
                "method": [
                        "train: real\nHP: varying",
                        "train: real\nHP: varying",
                        "train: synth.\nHP: varying",
                        "train: synth.\nHP: varying",
                        "train: synth.\nHP: fixed",
                        "train: synth.\nHP: fixed"
                        ],
                "means": [mean_train_steps[0], episode_num_needed_means[0],
                          mean_train_steps[1], episode_num_needed_means[1],
                          mean_train_steps[2], episode_num_needed_means[2]],
                "std dev": [std_train_steps[0], episode_num_needed_stds[0],
                            std_train_steps[1], episode_num_needed_stds[1],
                            std_train_steps[2], episode_num_needed_stds[2]]
                })

        scale = 1000
        barplot_df['means'] = np.where(barplot_df['type'] == 'episodes', barplot_df['means'] * scale, barplot_df['means'])
        barplot_df['std dev'] = np.where(barplot_df['type'] == 'episodes', barplot_df['std dev'] * scale, barplot_df['std dev'])

        p = barplot_err(x="method", y="means", yerr="std dev", hue="type", errwidth=1., capsize=.05, data=barplot_df, ax=axes[1, i],
                        palette=sns.color_palette("Paired"))

        axis_left = axes[1, i]
        axis_right = axis_left.twinx()
        # axis_right.set_ylim(axis_left.get_ylim())

        # axis_left.set_yticks(np.linspace(axis_left.get_ybound()[0], axis_left.get_ybound()[1], 5))
        # axis_right.set_yticks(np.linspace(axis_left.get_ybound()[0], axis_left.get_ybound()[1], 5))

        axis_right.set_yticks(np.linspace(axis_left.get_yticks()[0], axis_left.get_yticks()[-1], len(axis_left.get_yticks())))
        axis_left.set_yticks(np.linspace(axis_right.get_yticks()[0], axis_right.get_yticks()[-1], len(axis_right.get_yticks())))
        plt.setp(axis_right.get_yticklabels()[0], visible=False)
        plt.setp(axis_right.get_yticklabels()[-1], visible=False)

        for tick in axis_left.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)

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

        # todo:
        # for x, y, mean in zip([0, 1], [220, 220], mean_list):
        #     plt.text(x, y, mean, ha='center', va='top', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "AB_vary_hp_merged_plots.pdf"))
    plt.show()
