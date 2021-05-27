import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

from utils import barplot_err

FILE_DIRS = []
FILE_LISTS = []
PLOT_NAMES = []
TITLES = []
MEAN_TRAIN_STEPS = []
STD_TRAIN_STEPS = []

FILE_DIRS.append('/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole/ddqn_vary_trained_on')
TITLES.append("Trained synth. env. with DDQN")
FILE_LISTS.append(['0.pt', '2.pt', '1.pt'])

FILE_DIRS.append('/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole/ddqn_to_duelingddqn_vary')
TITLES.append("Transfer DDQN -> Dueling DDQN")
FILE_LISTS.append(['0.pt', '2.pt', '1.pt'])

FILE_DIRS.append('/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole' \
                 '/ddqn_to_td3_discrete_vary_td3HPs_variation_experiments/learned_temp_init_1_tanh_hard_True_lr_5e-4')
TITLES.append("Transfer DDQN -> TD3")
FILE_LISTS.append(['0.pt', '2.pt', '1.pt'])

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

show_5_best_jointly_with_other = True
show_only_kde = False

if show_5_best_jointly_with_other:
    # mode 2, 5 best models (see syn_env_evaluate_cartpole_vary_hp_2_TD3_discrete.py for which one), 4k evals (80(agents_num)*5(models)*10(
    # evals per model))
    td3_mean_train_steps[1] = 6287.5
    td3_std_train_steps[1] = 1970.6455160682756
    TITLES[2] = "Transfer DDQN -> TD3"
    FILE_LISTS[2] = ['0.pt', '2_5_best_filtered_models.pt', '1.pt']

if show_only_kde:
    plot_name = 'CP_vary_hp_merged_plots_best_5_dtd3_only_kde.pdf'
else:
    plot_name = 'CP_vary_hp_merged_plots_best_5_dtd3.pdf'

key = "2_5_best_filtered_models"  # don't comment this line
MEAN_TRAIN_STEPS.append(td3_mean_train_steps)
STD_TRAIN_STEPS.append(td3_std_train_steps)

if __name__ == "__main__":
    if show_only_kde:
        nrows = 1
        gridspec_kw = {}
        figsize = (15, 3)
    else:
        nrows = 2
        gridspec_kw = {'height_ratios': [2.3, 1.5]}
        figsize = (15, 5)

    fig, axes = plt.subplots(figsize=figsize, ncols=3, nrows=nrows, sharex="row", sharey="row", gridspec_kw=gridspec_kw)

    for i, data in enumerate(zip(FILE_DIRS, FILE_LISTS, TITLES, MEAN_TRAIN_STEPS, STD_TRAIN_STEPS)):
        FILE_DIR, FILE_LIST, title, mean_train_steps, std_train_steps = data

        data_list = []
        mean_list = []
        episode_num_needed_means = []
        episode_num_needed_stds = []
        reward_list_single_2 = []
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

                if show_5_best_jointly_with_other:
                    # show result from 2.pt jointly with 2_5_best_filtered_models.pt
                    save_dict = torch.load(os.path.join(FILE_DIR, "2.pt"))
                    reward_list_2 = save_dict['reward_list']

                    for r_list in reward_list_2:
                        reward_list_single_2 += r_list

                    data_list.append(reward_list_single_2)

            else:
                mean_episode_num = np.mean(save_dict["train_steps_needed"])
                std_episode_num = np.std(save_dict["train_steps_needed"])

            reward_list_single = []
            for r_list in reward_list:
                reward_list_single += r_list

            data_list.append(reward_list_single)
            if i == 0 and j == 0:
                mean_list.append('{:.2f}'.format((statistics.mean(reward_list_single))))
            elif key in file and show_5_best_jointly_with_other:
                mean_list.append('{:.2f} (all: {:.2f})'.format(statistics.mean(reward_list_single), statistics.mean(reward_list_single_2)))
            else:
                mean_list.append('{:.2f}'.format((statistics.mean(reward_list_single))))

            episode_num_needed_means.append(mean_episode_num)
            episode_num_needed_stds.append(std_episode_num)

        if len(data_list) == 4:
            data_dict = {
                    'train: real, HP: varying': data_list[0],

                    'train: synth., HP: varying': data_list[1],

                    'train: synth., HP: fixed': data_list[3],

                    'train: synth., HP: varying (5 best)': data_list[2],
                    }
        else:
            data_dict = {
                    'train: real, HP: varying': data_list[0],

                    'train: synth., HP: varying': data_list[1],

                    'train: synth., HP: fixed': data_list[2],
                    }

        df = pd.DataFrame(data=data_dict)
        df = df.melt(value_name="cumulative rewards", var_name="type")

        sns.set_context(rc={
                "font.size": 13,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "legend.fontsize": 11,
                "xtick.labelsize": 11,
                "ytick.labelsize": 9
                })

        if i == 0:
            title += " (CartPole-v0)"

        color_p = sns.color_palette()

        if i == 2 and show_5_best_jointly_with_other:
            palette = ["C0", "C1", "C2", "k"]
        else:
            palette = ["C0", "C1", "C2"]

        if show_only_kde:
            ax = axes[i]
        else:
            ax = axes[0, i]


        g = sns.kdeplot(x="cumulative rewards", hue="type", data=df, ax=ax, palette=palette)
        # sns.rugplot(x="cumulative rewards", hue="type", data=df, ax=ax, palette=palette)

        g.set_title(title)
        ax.xaxis.set_tick_params(labelsize=11)
        ax.yaxis.set_tick_params(labelsize=11)

        if i == 0:
            # remove legend title with hue-kdeplot
            g.axes.get_legend().set_title("")
            ax.set_xlabel('cumulative rewards')
            ax.set_ylabel('Density')
            if show_5_best_jointly_with_other:
                g.axes.get_legend().set_visible(False)

            ax2 = plt.axes([0.08, .75, .1, .1], facecolor='w')
            g2 = sns.kdeplot(x="cumulative rewards", hue="type", data=df, ax=ax2, palette=palette)
            g2.axes.get_legend().set_visible(False)
            ax2.set_title('zoom')
            ax2.get_xaxis().get_label().set_visible(False)
            ax2.get_yaxis().get_label().set_visible(False)
            ax2.set_yticks([])
            ax2.set_xlim([150, 200])
            ax2.set_ylim([0.0, 0.02])

        elif i == 2 and show_5_best_jointly_with_other:
            g.axes.get_legend().set_title("")
            ax.get_xaxis().get_label().set_visible(False)
            ax.get_yaxis().get_label().set_visible(False)

            # ax4 = plt.axes([.85, .65, .1, .1], facecolor='w')
            # g4 = sns.kdeplot(x="cumulative rewards", hue="type", data=df, ax=ax4, palette=palette)
            # g4.axes.get_legend().set_visible(False)
            # ax4.set_title('zoom')
            # ax4.get_xaxis().get_label().set_visible(False)
            # ax4.get_yaxis().get_label().set_visible(False)
            # ax4.set_yticks([])
            # ax4.set_xlim([50, 150])
            # ax4.set_ylim([0.0, 0.001])

        else:
            g.axes.get_legend().set_visible(False)
            ax.get_xaxis().get_label().set_visible(False)
            ax.get_yaxis().get_label().set_visible(False)

            # ax3 = plt.axes([.405, .75, .1, .1], facecolor='w')
            # g3 = sns.kdeplot(x="cumulative rewards", hue="type", data=df, ax=ax3, palette=palette)
            # g3.axes.get_legend().set_visible(False)
            # ax3.set_title('zoom')
            # ax3.get_xaxis().get_label().set_visible(False)
            # ax3.get_yaxis().get_label().set_visible(False)
            # ax3.set_yticks([])
            # ax3.set_xlim([50, 150])
            # ax3.set_ylim([0.0, 0.001])


        if not show_only_kde:

            barplot_data_dct = {
                    "type": ["steps", "episodes",
                             "steps", "episodes",
                             "steps", "episodes"],
                    "method": ["train: real\nHP: varying", "train: real\nHP: varying",
                               "train: synth.\nHP: varying", "train: synth.\nHP: varying",
                               "train: synth.\nHP: fixed", "train: synth.\nHP: fixed"],
                    "means": [mean_train_steps[0], episode_num_needed_means[0],
                              mean_train_steps[1], episode_num_needed_means[1],
                              mean_train_steps[2], episode_num_needed_means[2]],
                    "std dev": [std_train_steps[0], episode_num_needed_stds[0],
                                std_train_steps[1], episode_num_needed_stds[1],
                                std_train_steps[2], episode_num_needed_stds[2]]
                    }

            if show_5_best_jointly_with_other and i == 2:
                barplot_data_dct["method"] = ["train: real\nHP: varying", "train: real\nHP: varying",
                                              "train: synth.\nHP: varying (5 best)", "train: synth.\nHP: varying (5 best)",
                                              "train: synth.\nHP: fixed", "train: synth.\nHP: fixed"]
            else:
                barplot_data_dct["method"] = ["train: real\nHP: varying", "train: real\nHP: varying",
                                              "train: synth.\nHP: varying", "train: synth.\nHP: varying",
                                              "train: synth.\nHP: fixed", "train: synth.\nHP: fixed"]

            barplot_df = pd.DataFrame(barplot_data_dct)

            scale = 100
            barplot_df['means'] = np.where(barplot_df['type'] == 'episodes', barplot_df['means'] * scale, barplot_df['means'])
            barplot_df['std dev'] = np.where(barplot_df['type'] == 'episodes', barplot_df['std dev'] * scale, barplot_df['std dev'])

            p = barplot_err(x="method", y="means", yerr="std dev", hue="type", errwidth=1., capsize=.05, data=barplot_df, ax=axes[1, i],
                            palette=sns.color_palette("Paired"))

            axis_left = axes[1, i]
            axis_right = axis_left.twinx()
            axis_right.set_ylim(axis_left.get_ylim())
            axis_right.set_yticklabels(np.round(axis_left.get_yticks() / scale, 1).astype(int))

            if i == 0:
                p.axes.get_legend().set_title("")
                axis_left.set_ylabel("mean train steps", fontsize=11)
                axis_right.set_ylabel("mean train episodes", rotation=-90, fontsize=11, labelpad=12)
                axis_left.get_xaxis().get_label().set_visible(False)
                p.xaxis.set_tick_params(labelsize=11)

                if show_5_best_jointly_with_other:
                    axes[1, i].tick_params(labelbottom=True)

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
                p.xaxis.set_tick_params(labelsize=11)
                if show_5_best_jointly_with_other:
                    axes[1, i].tick_params(labelbottom=True)

            for x, y, mean in zip([0, 1, 2], [220, 220, 220], mean_list):
                plt.text(x, y, mean, ha='center', va='top', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "transfer_experiments/cartpole", plot_name))
    plt.show()

