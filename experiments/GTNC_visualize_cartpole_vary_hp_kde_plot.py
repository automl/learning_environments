import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

FILE_DIRS = []
FILE_LISTS = []
PLOT_NAMES = []
TITLES = []
MEAN_TRAIN_STEPS = []
STD_TRAIN_STEPS = []

"""
0.pt -> rewards train: real
2.pt -> rewards train: synth., HPs: varied
1.pt -> rewards train: synth., HPs: fixed
NEW: 3.pt -> rewards from MBRL baseline
"""

FILE_DIRS.append('/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole/ddqn_vary_trained_on')
TITLES.append("DDQN on DDQN-trained SEs")
FILE_LISTS.append(['0.pt', '2.pt', '1.pt'])

FILE_DIRS.append('/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole/ddqn_to_duelingddqn_vary')
TITLES.append("Transfer: Dueling DDQN on DDQN-trained SEs")
FILE_LISTS.append(['0.pt', '2.pt', '1.pt'])

FILE_DIRS.append('/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole' \
                 '/ddqn_to_td3_discrete_vary_td3HPs_variation_experiments/learned_temp_init_1_tanh_hard_True_lr_5e-4')
TITLES.append("Transfer: TD3 on DDQN-trained SEs")
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

############ PARAMETERS ############
show_5_best_jointly_with_other = True
show_zoom = False

if show_5_best_jointly_with_other:
    # mode 2, 5 best models (see syn_env_evaluate_cartpole_vary_hp_2_TD3_discrete.py for which one), 4k evals (80(agents_num)*5(models)*10(
    # evals per model))
    td3_mean_train_steps[1] = 6287.5
    td3_std_train_steps[1] = 1970.6455160682756
    TITLES[2] = "Transfer: TD3 on DDQN-trained SEs"
    FILE_LISTS[2] = ['0.pt', '2_5_best_filtered_models.pt', '1.pt']

if show_zoom:
    plot_name = 'CP_vary_hp_merged_plots_best_5_dtd3_only_kde_zoom.pdf'
else:
    plot_name = 'CP_vary_hp_merged_plots_best_5_dtd3_only_kde.pdf'

key = "2_5_best_filtered_models"  # don't comment this line
MEAN_TRAIN_STEPS.append(td3_mean_train_steps)
STD_TRAIN_STEPS.append(td3_std_train_steps)

if __name__ == "__main__":

    nrows = 1
    gridspec_kw = {}
    figsize = (15, 3)

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
                "ytick.labelsize": 9,
                })

        if i == 0:
            title += " (CartPole-v0)"

        color_p = sns.color_palette()

        if i == 2 and show_5_best_jointly_with_other:
            palette = ["C0", "C1", "C2", "k"]
        else:
            palette = ["C0", "C1", "C2"]

        ax = axes[i]

        # def upper_rugplot(data, ax=None, i=None, color=None, **kwargs):
        #     from matplotlib.collections import LineCollection
        #     ax = ax or plt.gca()
        #     heights = [[0.0, 0.03], [0.05, 0.08], [0.1, 0.13]]
        #
        #     height = heights[i]
        #     kwargs.setdefault("linewidth", 0.2)
        #     segs = np.stack((np.c_[data, data],
        #                      np.c_[np.ones_like(data) - height[0], np.ones_like(data) - height[1]]),
        #                     axis=-1)
        #     lc = LineCollection(segs, transform=ax.get_xaxis_transform(), color=color, **kwargs)
        #     ax.add_collection(lc)
        #
        # l_colors = ['C0', 'C1', 'C2']
        # for k, typ in enumerate(["train: real", "train: synth., HPs: varied", "train: synth., HPs: fixed"]):
        #     data = df[df['type'].str.contains(typ)]['cumulative rewards']
        #     if i == 2 and k == 1:
        #         data = df[df['type'].str.contains("5 best")]['cumulative rewards']
        #         color = 'black'
        #     else:
        #         color = l_colors[k]
        #     upper_rugplot(data=data, ax=ax, i=k, color=color)

        # need to set label for accesss to handles later on
        g = sns.kdeplot(x="cumulative rewards", hue="type", data=df, ax=ax, palette=palette, label=data_dict.keys())

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

            if show_zoom:
                ax2 = plt.axes([0.11, .65, .08, .1], facecolor='w')
                l1 = g.get_lines()[0]
                l2 = g.get_lines()[1]
                l3 = g.get_lines()[2]
                ax2.plot(l1.get_data()[0], l1.get_data()[1], color=l1.get_color())
                ax2.plot(l2.get_data()[0], l2.get_data()[1], color=l2.get_color())
                ax2.plot(l3.get_data()[0], l3.get_data()[1], color=l3.get_color())
                ax2.set_title('zoom')
                ax2.get_xaxis().get_label().set_visible(False)
                ax2.get_yaxis().get_label().set_visible(False)
                ax2.set_yticks([0.0, 0.0005])
                ax2.set_xlim([50, 100])
                ax2.set_xticks([50, 75, 100])
                # ax2.set_ylim(g.get_ylim())
                ax2.set_ylim([0.0, 0.0005])

        elif i == 2 and show_5_best_jointly_with_other:
            g.axes.get_legend().set_title("")
            ax.get_xaxis().get_label().set_visible(False)
            ax.get_yaxis().get_label().set_visible(False)

            if show_zoom:
                g.axes.get_legend().set_visible(False)
                leg = plt.legend(reversed(g.axes.get_legend_handles_labels()[0]), data_dict.keys(), bbox_to_anchor=(-2.8, -6.5, 8, 1),
                                 mode="expand", ncol=4, frameon=False)

                for legobj in leg.legendHandles:
                    legobj.set_linewidth(2.0)


                ax4 = plt.axes([0.755, .65, .08, .1], facecolor='w')
                l1 = g.get_lines()[0]
                l2 = g.get_lines()[1]
                l3 = g.get_lines()[2]
                l4 = g.get_lines()[3]
                ax4.plot(l1.get_data()[0], l1.get_data()[1], color=l1.get_color())
                ax4.plot(l2.get_data()[0], l2.get_data()[1], color=l2.get_color())
                ax4.plot(l3.get_data()[0], l3.get_data()[1], color=l3.get_color())
                ax4.plot(l4.get_data()[0], l4.get_data()[1], color=l4.get_color())
                # ax4.set_title('zoom')
                ax4.get_xaxis().get_label().set_visible(False)
                ax4.get_yaxis().get_label().set_visible(False)
                ax4.set_yticks([0.0, 0.001])
                ax4.set_xlim([50, 100])
                ax4.set_xticks([50, 75, 100])
                ax4.set_ylim([0.0, 0.001])

        else:
            g.axes.get_legend().set_visible(False)
            ax.get_xaxis().get_label().set_visible(False)
            ax.get_yaxis().get_label().set_visible(False)

            if show_zoom:
                ax3 = plt.axes([0.435, .65, .08, .1], facecolor='w')
                l1 = g.get_lines()[0]
                l2 = g.get_lines()[1]
                l3 = g.get_lines()[2]
                ax3.plot(l1.get_data()[0], l1.get_data()[1], color=l1.get_color())
                ax3.plot(l2.get_data()[0], l2.get_data()[1], color=l2.get_color())
                ax3.plot(l3.get_data()[0], l3.get_data()[1], color=l3.get_color())
                # ax3.set_title('zoom')
                ax3.get_xaxis().get_label().set_visible(False)
                ax3.get_yaxis().get_label().set_visible(False)
                ax3.set_yticks([0.0, 0.0005])
                ax3.set_xlim([50, 100])
                ax3.set_xticks([50, 75, 100])
                ax3.set_ylim([0.0, 0.0005])

    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "transfer_experiments/cartpole", plot_name))
    plt.show()
