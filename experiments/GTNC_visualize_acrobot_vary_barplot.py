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

FILE_DIRS.append('/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/acrobot/' \
                 '/ddqn_vary_trained_on')
TITLES.append("DDQN on DDQN-trained SE")
FILE_LISTS.append(['0.pt', '2.pt', '1.pt'])

FILE_DIRS.append('/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/acrobot' \
                 '/ddqn_to_duelingddqn_vary')
TITLES.append("Transfer DDQN -> Dueling DDQN")
FILE_LISTS.append(['0.pt', '2.pt', '1.pt'])

FILE_DIRS.append('/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/acrobot/ddqn_to_td3_discrete_vary/td3_discrete_vary_layer_norm_2_config')
TITLES.append("Transfer DDQN -> TD3")
FILE_LISTS.append(['0.pt', '2.pt', '1.pt'])

plot_name = "AB_vary_hp_merged_plots.pdf"

ddqn_mean_train_steps = [15388.76, 18240.325, 18177.8325]
ddqn_std_train_steps = [12820.303525556637, 6738.70980896, 8802.080576457]

MEAN_TRAIN_STEPS.append(ddqn_mean_train_steps)
STD_TRAIN_STEPS.append(ddqn_std_train_steps)

dueling_ddqn_mean_train_steps = [29531.1275, 18376.2125, 18540.66]
dueling_ddqn_std_train_steps = [79218.56009538576, 6401.7216619706105, 8945.47788686552]

MEAN_TRAIN_STEPS.append(dueling_ddqn_mean_train_steps)
STD_TRAIN_STEPS.append(dueling_ddqn_std_train_steps)

#  ddqn_to_td3_discrete_vary_layer_norm_2_learned_temp
td3_mean_train_steps = [80837.0275, 14138.3325, 13481.41]
td3_std_train_steps = [100188.32419489679, 3994.484271084, 5739.58887176599]

show_5_best_jointly_with_other = True

if show_5_best_jointly_with_other:
    # mode 2, 5 best models (see syn_env_evaluate_cartpole_vary_hp_2_TD3_discrete.py for which one), 4k evals (80(agents_num)*5(models)*10(
    # evals per model))
    td3_mean_train_steps[1] = 14408.2775
    td3_std_train_steps[1] = 3192.696929007473
    TITLES[2] = "Transfer DDQN -> TD3"
    FILE_LISTS[2] = ['0.pt', '2_5_best_filtered_models.pt', '1.pt']

plot_name = 'AB_vary_hp_merged_plots_best_5_dtd3_only_barplot.pdf'

key = "2_5_best_filtered_models"  # don't comment this line
MEAN_TRAIN_STEPS.append(td3_mean_train_steps)
STD_TRAIN_STEPS.append(td3_std_train_steps)

if __name__ == "__main__":
    nrows = 1
    gridspec_kw = {}
    figsize = (15, 2.3)

    fig, axes = plt.subplots(figsize=figsize, ncols=3, nrows=nrows, sharex=False, sharey=False, gridspec_kw=gridspec_kw)

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

            elif "td3_discrete" in file_path:
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
                mean_list.append('{:.2f}'.format((statistics.mean(reward_list_single))))
            elif key in file and show_5_best_jointly_with_other:
                mean_list.append('{:.2f} (all: {:.2f})'.format(statistics.mean(reward_list_single), statistics.mean(reward_list_single_2)))
            else:
                mean_list.append('{:.2f}'.format((statistics.mean(reward_list_single))))

            episode_num_needed_means.append(mean_episode_num)
            episode_num_needed_stds.append(std_episode_num)

        if len(data_list) == 4:
            data_dict = {
                'train: real': data_list[0],

                'train: synth., HPs: varied': data_list[1],

                'train: synth., HPs: fixed': data_list[3],

                'train: synth., HPs: varied (5 best)': data_list[2],
            }
        else:
            data_dict = {
                'train: real': data_list[0],

                'train: synth., HPs: varied': data_list[1],

                'train: synth., HPs: fixed': data_list[2],
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
            title += " (Acrobot-v1)"

        color_p = sns.color_palette()

        if i == 2 and show_5_best_jointly_with_other:
            palette = ["C0", "C1", "C2", "k"]
        else:
            palette = ["C0", "C1", "C2"]

        barplot_data_dct = {
            "type": ["steps", "episodes",
                     "steps", "episodes",
                     "steps", "episodes"],
            "method": ["train: real", "train: real",
                       "train: synth.\nHPs: varied", "train: synth.\nHPs: varied",
                       "train: synth.\nHPs: fixed", "train: synth.\nHPs: fixed"],
            "means": [mean_train_steps[0], episode_num_needed_means[0],
                      mean_train_steps[1], episode_num_needed_means[1],
                      mean_train_steps[2], episode_num_needed_means[2]],
            "std dev": [std_train_steps[0], episode_num_needed_stds[0],
                        std_train_steps[1], episode_num_needed_stds[1],
                        std_train_steps[2], episode_num_needed_stds[2]]
        }

        if show_5_best_jointly_with_other and i == 2:
            barplot_data_dct["method"] = ["train: real", "train: real",
                                          "train: synth.\nHPs: varied\n(5 best)", "train: synth.\nHPs: varied\n(5 best)",
                                          "train: synth.\nHPs: fixed", "train: synth.\nHPs: fixed"]
        else:
            barplot_data_dct["method"] = ["train: real", "train: real",
                                          "train: synth.\nHPs: varied", "train: synth.\nHPs: varied",
                                          "train: synth.\nHPs: fixed", "train: synth.\nHPs: fixed"]

        barplot_df = pd.DataFrame(barplot_data_dct)

        scale = 1000
        barplot_df['means'] = np.where(barplot_df['type'] == 'episodes', barplot_df['means'] * scale, barplot_df['means'])
        barplot_df['std dev'] = np.where(barplot_df['type'] == 'episodes', barplot_df['std dev'] * scale, barplot_df['std dev'])

        p = barplot_err(x="method", y="means", yerr="std dev", hue="type", errwidth=1., capsize=.05, data=barplot_df, ax=axes[i],
                        palette=sns.color_palette("Paired"))

        axis_left = axes[i]
        # axis_right = axis_left.twinx()
        # axis_right.set_ylim(axis_left.get_ylim())
        # axis_right.set_yticklabels(np.round(axis_left.get_yticks() / scale, 1).astype(int))

        if i == 0:
            p.axes.get_legend().set_title("")
            axis_left_after = axes[i]
            # axis_left_after.set_ylim((-10000, 100000))
            axis_left_after.set_ylim((-40000, axis_left_after.get_ylim()[1]))
            axis_right_after = axis_left.twinx()
            axis_left_after.set_ylabel("mean train steps", fontsize=11)
            axis_right_after.set_ylabel("mean train episodes", rotation=-90, fontsize=11, labelpad=12)
            axis_left.get_xaxis().get_label().set_visible(False)
            p.xaxis.set_tick_params(labelsize=11)

            if show_5_best_jointly_with_other:
                axes[i].tick_params(labelbottom=True)

        else:
            p.axes.get_xaxis().get_label().set_visible(False)
            p.axes.get_legend().set_visible(False)

            axis_left = axes[i]
            # axis_left.set_ylim((-10000, 100000))
            axis_left.set_ylim((-40000, axis_left.get_ylim()[1]))
            axis_right = axis_left.twinx()
            axis_right.set_ylim(axis_left.get_ylim())
            axis_right.set_yticklabels(np.round(axis_left.get_yticks() / scale, 1).astype(int))

            # remove tick labels from 2nd and 3rd plot but keep it in first while sharedx is active
            axes[i].tick_params(labelbottom=False)

            axis_left.get_yaxis().get_label().set_visible(False)
            axis_left.get_xaxis().get_label().set_visible(False)
            axis_right.get_yaxis().get_label().set_visible(False)
            # axis_right.get_yaxis().set_ticklabels([])
            axes[i].get_xaxis().get_label().set_visible(False)
            p.xaxis.set_tick_params(labelsize=11)

            if show_5_best_jointly_with_other:
                axes[i].tick_params(labelbottom=True)

        for x, y, mean in zip([0, 1, 2], [220, 220, 220], mean_list):
            plt.text(x, y, mean, ha='center', va='top', fontsize=10.5)

        axis_right_after.set_ylim(axis_left_after.get_ylim())
        axis_right_after.set_yticklabels(np.round(axis_left_after.get_yticks() / scale, 1).astype(int))

    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "transfer_experiments/acrobot", plot_name))
    plt.show()
