import os

import matplotlib.pyplot as plt
import numpy as np
import torch

LOG_FILES = [
             '../results/halfcheetah_compare_reward_envs/best_transfer_algo1.pt',
             '../results/halfcheetah_compare_reward_envs/best_transfer_algo2.pt',
             '../results/halfcheetah_compare_reward_envs/best_transfer_algo5.pt',
             '../results/halfcheetah_compare_reward_envs/best_transfer_algo6.pt',

             '../results/halfcheetah_compare_reward_envs/best_transfer_algo0.pt',
             '../results/halfcheetah_compare_reward_envs/best_transfer_algo-1.pt',  # todo: BOHB

             '../results/halfcheetah_compare_reward_envs/best_transfer_algo3.pt',
             '../results/halfcheetah_compare_reward_envs/best_transfer_algo4.pt',
             '../results/halfcheetah_compare_reward_envs/best_transfer_algo7.pt',
             '../results/halfcheetah_compare_reward_envs/best_transfer_algo8.pt',

             '../results/halfcheetah_compare_reward_envs/best_transfer_algo101.pt',
             '../results/halfcheetah_compare_reward_envs/best_transfer_algo102.pt'
             ]

# LOG_FILES = [
#              '../results/halfcheetah_compare_reward_envs_lr_1e-5/best_transfer_algo1.pt',
#              '../results/halfcheetah_compare_reward_envs_lr_1e-5/best_transfer_algo2.pt',
#              '../results/halfcheetah_compare_reward_envs_lr_1e-5/best_transfer_algo5.pt',
#              '../results/halfcheetah_compare_reward_envs_lr_1e-5/best_transfer_algo6.pt',
#
#              '../results/halfcheetah_compare_reward_envs_lr_1e-5/best_transfer_algo0.pt',
#              '../results/halfcheetah_compare_reward_envs_lr_1e-5/best_transfer_algo-1.pt',
#
#             '../results/halfcheetah_compare_reward_envs_lr_1e-5/best_transfer_algo3.pt',
#             '../results/halfcheetah_compare_reward_envs_lr_1e-5/best_transfer_algo4.pt',
#             '../results/halfcheetah_compare_reward_envs_lr_1e-5/best_transfer_algo7.pt',
#             '../results/halfcheetah_compare_reward_envs_lr_1e-5/best_transfer_algo8.pt',
#
#              '../results/halfcheetah_compare_reward_envs_lr_1e-5/best_transfer_algo101.pt',
#              '../results/halfcheetah_compare_reward_envs_lr_1e-5/best_transfer_algo102.pt'
#              ]

# LOG_FILES = [
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo1.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo2.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo5.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo6.pt',
#
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo0.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo-1.pt',
#
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo3.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo4.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo7.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo8.pt',
#
#         '../results/halfcheetah_compare_reward_envs/best_transfer_algo101.pt',  # running
#         '../results/halfcheetah_compare_reward_envs/best_transfer_algo102.pt'  # running
#         ]

# ent_coef or vf_coef important?
# LOG_FILES = [
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo1_2.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo2_2.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo5_2.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo6_2.pt',
#
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo0_2.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo-1_2.pt',
#
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo3_2.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo4_2.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo7_2.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo8_2.pt',
#
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo101_2.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo102_2.pt'
#         ]


# LOG_FILES = [
        # '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo1_3.pt',
        # '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo2_3.pt',
        # '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo5_3.pt',
        # '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo6_3.pt',
        #
        # '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo0_3.pt',
        # '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo-1_3.pt',
        #
        # '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo3_3.pt',
        # '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo4_3.pt',
        # '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo7_3.pt',
        # '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo8_3.pt',
        #
        # '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo101_3.pt',
        # '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo102_3.pt'
        # ]

# LOG_FILES = [
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo1_4.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo2_4.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo5_4.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo6_4.pt',
#
        # '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo0_4.pt',
        # '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo-1_4.pt',
#
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo3_4.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo4_4.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo7_4.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo8_4.pt',
#
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo101_4.pt',
#         '../results/halfcheetah_compare_reward_envs_bohb_opt_1k_episodes/best_transfer_algo102_4.pt'
#         ]


# 0.001 ent_coeff, 0.1 action_std
LOG_FILES = [
        '../results/halfcheetah_compare_reward_envs_3k_episodes_1e-3_ent_coef_1e-1_action_std/best_transfer_algo1.pt',
        '../results/halfcheetah_compare_reward_envs_3k_episodes_1e-3_ent_coef_1e-1_action_std/best_transfer_algo2.pt',
        '../results/halfcheetah_compare_reward_envs_3k_episodes_1e-3_ent_coef_1e-1_action_std/best_transfer_algo5.pt',
        '../results/halfcheetah_compare_reward_envs_3k_episodes_1e-3_ent_coef_1e-1_action_std/best_transfer_algo6.pt',

        '../results/halfcheetah_compare_reward_envs_3k_episodes_1e-3_ent_coef_1e-1_action_std/best_transfer_algo0.pt',
        '../results/halfcheetah_compare_reward_envs_3k_episodes_1e-3_ent_coef_1e-1_action_std/best_transfer_algo-1.pt',

        '../results/halfcheetah_compare_reward_envs_3k_episodes_1e-3_ent_coef_1e-1_action_std/best_transfer_algo3.pt',
        '../results/halfcheetah_compare_reward_envs_3k_episodes_1e-3_ent_coef_1e-1_action_std/best_transfer_algo4.pt',
        '../results/halfcheetah_compare_reward_envs_3k_episodes_1e-3_ent_coef_1e-1_action_std/best_transfer_algo7.pt',
        '../results/halfcheetah_compare_reward_envs_3k_episodes_1e-3_ent_coef_1e-1_action_std/best_transfer_algo8.pt',

        '../results/halfcheetah_compare_reward_envs_3k_episodes_1e-3_ent_coef_1e-1_action_std/best_transfer_algo101.pt',
        '../results/halfcheetah_compare_reward_envs_3k_episodes_1e-3_ent_coef_1e-1_action_std/best_transfer_algo102.pt'
        ]


LEGEND = [
        'PPO + exc. pot. RN',
        'PPO + add. pot. RN',
        'PPO + exc. non-pot. RN',
        'PPO + add. non-pot. RN',

        'PPO',
        'PPO + ICM',

        'PPO + exc. pot. RN + augm.',
        'PPO + add. pot. RN + augm.',
        'PPO + exc. non-pot. RN + augm.',
        'PPO + add. non-pot. RN + augm.',

        'PPO + exc. ER',
        'PPO + add. ER',
        ]

STD_MULT = 0.2
MIN_STEPS = 3000000


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

    for i, data in enumerate(proc_data):
        mean, std = data
        if i == 10:
            plt.plot(mean, color='#575757', linewidth=.5)
        elif i == 11:
            plt.plot(mean, color='#EBBB00', linewidth=.5)
        else:
            plt.plot(mean, linewidth=.5)

    for i, data in enumerate(proc_data):
        mean, std = data
        if i == 10:
            plt.fill_between(x=range(len(mean)), y1=mean - std * STD_MULT, y2=mean + std * STD_MULT, alpha=0.1, color='#575757')
        elif i == 11:
            plt.fill_between(x=range(len(mean)), y1=mean - std * STD_MULT, y2=mean + std * STD_MULT, alpha=0.1, color='#EBBB00')
        else:
            plt.fill_between(x=range(len(mean)), y1=mean - std * STD_MULT, y2=mean + std * STD_MULT, alpha=0.1)

    plt.legend(LEGEND, fontsize=7)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.title('HalfCheetah-v3 Transfer')
    plt.xlabel('steps')
    plt.xlim(0, MIN_STEPS)
    plt.ylim(-1000, 6000)
    plt.ylabel('cumulative reward')
    base_dir = os.path.dirname(LOG_FILES[0])
    plt.savefig(os.path.join(base_dir, savefig_name))
    plt.show()


if __name__ == "__main__":
    proc_data = get_data()
    # plot_data(proc_data=proc_data, savefig_name=f'halfcheetah_transfer_algo.pdf')
    plot_data(proc_data=proc_data, savefig_name=f'halfcheetah_transfer_algo_bohb_1k_eps.png')
