import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# FILE_DIR = '/home/dingsda/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_vary_hp_2020-11-17-10/GTN_models_CartPole-v0'

FILE_DIRS = []
PLOT_NAMES = []
TITLES = []
MEAN_TRAIN_STEPS = []
STD_TRAIN_STEPS = []

FILE_DIRS.append('/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole/ddqn_vary_trained_on')
PLOT_NAMES.append('cartpole_ddqn_vary_hp_both_lengths.eps')
TITLES.append("Trained synth. env. with DDQN")

FILE_DIRS.append('/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole/ddqn_to_duelingddqn_vary')
PLOT_NAMES.append('cartpole_ddqn_to_duelingddqn_vary_hp_both_lengths.eps')
TITLES.append("Transfer DDQN -> Dueling DDQN")

# FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole/ddqn_to_td3_discrete_vary'
# plot_name = 'cartpole_ddqn_to_td3_discrete_vary_hp_both_lengths.eps'
# title = "Transfer DDQN -> TD3"

# FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole
# /ddqn_to_td3_discrete_vary_td3HPs_variation_experiments/not_learned_init_1_relu_hard_False_lr_1e-3'
# plot_name = 'ddqn_to_td3_discrete_gumbel_not_learned_temp_relu.eps'
# title = "Transfer DDQN -> TD3"

# FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole
# /ddqn_to_td3_discrete_vary_td3HPs_variation_experiments/learned_temp_init_1_relu_hard_False_lr_5e-4'
# plot_name = 'ddqn_to_td3_discrete_gumbel_learned_temp_relu_hard_false_lr_5e-4.eps'
# title = "Transfer DDQN -> TD3"

FILE_DIRS.append('/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole' \
                 '/ddqn_to_td3_discrete_vary_td3HPs_variation_experiments/learned_temp_init_1_tanh_hard_True_lr_5e-4')
PLOT_NAMES.append('ddqn_to_td3_discrete_gumbel_learned_temp_tanh_both_lengths.eps')
TITLES.append("Transfer DDQN -> TD3")

# FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole
# /ddqn_to_td3_discrete_vary_td3HPs_variation_experiments/learned_temp_tanh_config'
# plot_name = 'ddqn_to_td3_discrete_gumbel_learned_temp_tanh_config.eps'
# title = "Transfer DDQN -> TD3"

# FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole
# /ddqn_to_td3_discrete_vary_td3HPs_variation_experiments/learned_temp_relu_layer_norm'
# plot_name = 'ddqn_to_td3_discrete_gumbel_learned_temp_relu_layer_norm.eps'
# title = "Transfer DDQN -> TD3"

# FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole
# /ddqn_to_td3_discrete_vary_td3HPs_variation_experiments/learned_temp_better_config'
# plot_name = 'ddqn_to_td3_discrete_gumbel_learned_temp_better_config.eps'
# title = "Transfer DDQN -> TD3"

# FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole
# /ddqn_to_td3_discrete_vary_td3HPs_variation_experiments/learned_temp_tanh_less_hpo_config'
# plot_name = 'ddqn_to_td3_discrete_gumbel_learned_temp_tanh_less_hpo_config.eps'
# title = "Transfer DDQN -> TD3"

# FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole
# /ddqn_to_td3_discrete_vary_td3HPs_variation_experiments/learned_temp_tanh_less_hpo_layer_norm_config'
# plot_name = 'ddqn_to_td3_discrete_gumbel_learned_temp_tanh_less_hpo_layer_norm_config.eps'
# title = "Transfer DDQN -> TD3"

# FILE_DIR = '/home/ferreira/Projects/learning_environments/experiments/transfer_experiments/cartpole/correct_episode_length
# /ddqn_vary_trained_on'
# plot_name = 'cartpole_ddqn_vary_hp_episode_length.eps'
# title = "Trained synth. env. with DDQN"

FILE_LIST = ['0.pt', '2.pt', '1.pt']
# FILE_LIST = ['0.pt', '1.pt']


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

MEAN_TRAIN_STEPS.append(td3_mean_train_steps)
STD_TRAIN_STEPS.append(td3_std_train_steps)

# mean_train_steps = td3_mean_train_steps
# std_train_steps = td3_std_train_steps


def barplot_err(x, y, xerr=None, yerr=None, data=None, **kwargs):

    _data = []
    for _i in data.index:

        _data_i = pd.concat([data.loc[_i:_i]]*3, ignore_index=True, sort=False)
        _row = data.loc[_i]
        if xerr is not None:
            _data_i[x] = [_row[x]-_row[xerr], _row[x], _row[x]+_row[xerr]]
        if yerr is not None:
            _data_i[y] = [_row[y]-_row[yerr], _row[y], _row[y]+_row[yerr]]
        _data.append(_data_i)

    _data = pd.concat(_data, ignore_index=True, sort=False)

    _ax = sns.barplot(x=x,y=y,data=_data,ci='sd',**kwargs)

    return _ax


if __name__ == "__main__":
    # fig = plt.figure(figsize=(22.5, 9))
    fig, axes = plt.subplots(figsize=(16, 5), ncols=3, nrows=2, sharex="row", sharey="row", gridspec_kw={'height_ratios': [2, 1.2]})

    for i, data in enumerate(zip(FILE_DIRS, PLOT_NAMES, TITLES, MEAN_TRAIN_STEPS, STD_TRAIN_STEPS)):
        FILE_DIR, plot_name, title, mean_train_steps, std_train_steps = data

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
            mean_list.append('mean reward: {:.2f}'.format((statistics.mean(reward_list_single))))

            episode_num_needed_means.append(mean_episode_num)
            episode_num_needed_stds.append(std_episode_num)



        data_dict = {
                'train: real, HP: varying': data_list[0],

                'train: synth., HP: varying': data_list[1],

                'train: synth., HP: fixed': data_list[2]}

        df = pd.DataFrame(data=data_dict)
        df = df.melt(value_name="cumulative rewards", var_name="type")

        sns.set_context(rc={
                "font.size": 8.5,
                "axes.titlesize": 10,
                "axes.labelsize": 10
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

        p = barplot_err(x="method", y="means", yerr="std dev", hue="type", errwidth=1., capsize=.05, data=barplot_df, ax=axes[1,i],
                        palette=sns.color_palette("Paired"))

        axis_left = axes[1, i]
        axis_right = axis_left.twinx()
        axis_right.set_ylim(axis_left.get_ylim())
        axis_right.set_yticklabels(np.round(axis_left.get_yticks() / scale, 1).astype(int))

        if i == 0:
            p.axes.get_legend().set_title("")
            axis_left.set_ylabel("mean train steps")
            axis_right.set_ylabel("mean train episodes", rotation=-90, labelpad=10)
            axis_left.get_xaxis().get_label().set_visible(False)

            # x = p.patches[0].get_height()
            # y = p.patches[0].get_x()
            # p.annotate("test", (y, x+9.), color="black", va="bottom")

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
        for x, y, mean in zip([0, 1, 2], [220, 220, 220], mean_list):
            plt.text(x, y, mean, ha='center', va='top')

    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "CP_vary_hp_merged_plots.pdf"))
    plt.show()
