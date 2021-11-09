import os

import matplotlib.pyplot as plt
import numpy as np
import torch

AUC = True

if AUC:
    # 0.001 ent_coeff, 0.1 action_std
    LOG_FILES = [
        '../results/3_rn_auc/halfcheetah_compare_reward_envs/best_transfer_algo1.pt',
        '../results/3_rn_auc/halfcheetah_compare_reward_envs/best_transfer_algo2.pt',
        '../results/3_rn_auc/halfcheetah_compare_reward_envs/best_transfer_algo5.pt',
        '../results/3_rn_auc/halfcheetah_compare_reward_envs/best_transfer_algo6.pt',

        '../results/0_before_auc/halfcheetah_compare_reward_envs_5k_episodes_1e-3_ent_coeff_1e-1_action_std/best_transfer_algo0.pt',
        '../results/0_before_auc/halfcheetah_compare_reward_envs_5k_episodes_1e-3_ent_coeff_1e-1_action_std/best_transfer_algo-1.pt',

        '../results/3_rn_auc/halfcheetah_compare_reward_envs/best_transfer_algo3.pt',
        '../results/3_rn_auc/halfcheetah_compare_reward_envs/best_transfer_algo4.pt',
        '../results/3_rn_auc/halfcheetah_compare_reward_envs/best_transfer_algo7.pt',
        '../results/3_rn_auc/halfcheetah_compare_reward_envs/best_transfer_algo8.pt',

        # '../results/3_rn_auc/halfcheetah_compare_reward_envs/best_transfer_algo101.pt',
        # '../results/3_rn_auc/halfcheetah_compare_reward_envs/best_transfer_algo102.pt'
    ]

else:
    # 0.001 ent_coeff, 0.1 action_std
    LOG_FILES = [
        '../results/0_before_auc/halfcheetah_compare_reward_envs_5k_episodes_1e-3_ent_coeff_1e-1_action_std/best_transfer_algo1.pt',
        '../results/0_before_auc/halfcheetah_compare_reward_envs_5k_episodes_1e-3_ent_coeff_1e-1_action_std/best_transfer_algo2.pt',
        '../results/0_before_auc/halfcheetah_compare_reward_envs_5k_episodes_1e-3_ent_coeff_1e-1_action_std/best_transfer_algo5.pt',
        '../results/0_before_auc/halfcheetah_compare_reward_envs_5k_episodes_1e-3_ent_coeff_1e-1_action_std/best_transfer_algo6.pt',

        '../results/0_before_auc/halfcheetah_compare_reward_envs_5k_episodes_1e-3_ent_coeff_1e-1_action_std/best_transfer_algo0.pt',
        '../results/0_before_auc/halfcheetah_compare_reward_envs_5k_episodes_1e-3_ent_coeff_1e-1_action_std/best_transfer_algo-1.pt',

        '../results/0_before_auc/halfcheetah_compare_reward_envs_5k_episodes_1e-3_ent_coeff_1e-1_action_std/best_transfer_algo3.pt',
        '../results/0_before_auc/halfcheetah_compare_reward_envs_5k_episodes_1e-3_ent_coeff_1e-1_action_std/best_transfer_algo4.pt',
        '../results/0_before_auc/halfcheetah_compare_reward_envs_5k_episodes_1e-3_ent_coeff_1e-1_action_std/best_transfer_algo7.pt',
        '../results/0_before_auc/halfcheetah_compare_reward_envs_5k_episodes_1e-3_ent_coeff_1e-1_action_std/best_transfer_algo8.pt',

        # '../results/0_before_auc/halfcheetah_compare_reward_envs_5k_episodes_1e-3_ent_coeff_1e-1_action_std/best_transfer_algo101.pt',
        # '../results/0_before_auc/halfcheetah_compare_reward_envs_5k_episodes_1e-3_ent_coeff_1e-1_action_std/best_transfer_algo102.pt'
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

    # 'PPO + exc. ER',
    # 'PPO + add. ER',
]

STD_MULT = 0.2
MIN_STEPS = 5000000


# MIN_STEPS = 2000000


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

    from scipy.interpolate import make_interp_spline

    for i, data in enumerate(proc_data):
        mean, std = data
        x = np.linspace(1, len(mean), num=len(mean))
        xnew = np.linspace(1, len(mean), num=300)
        spl = make_interp_spline(x, mean, k=3)
        mean_smooth = spl(xnew)

        if i == 10:
            # plt.plot(mean, color='#575757', linewidth=.8)
            plt.plot(xnew, mean_smooth, color='#575757', linewidth=1)
        elif i == 11:
            # plt.plot(mean, color='#EBBB00', linewidth=.8)
            plt.plot(xnew, mean_smooth, color='#EBBB00', linewidth=1)
        else:
            # plt.plot(mean, linewidth=.8)
            plt.plot(xnew, mean_smooth, linewidth=1)

    for i, data in enumerate(proc_data):
        mean, std = data

        # x = np.linspace(1, len(mean), num=len(mean))
        # xnew = np.linspace(1, len(mean), num=300)
        # spl = make_interp_spline(x, mean, k=3)
        # mean_smooth = spl(xnew)
        #
        # spl_std = make_interp_spline(x, std, k=3)
        # std_smooth = spl_std(xnew)

        if i == 10:
            plt.fill_between(x=range(len(mean)), y1=mean - std * STD_MULT, y2=mean + std * STD_MULT, alpha=0.2, color='#575757')
            # plt.fill_between(x=range(len(mean)), y1=mean - std * STD_MULT, y2=mean + std * STD_MULT, alpha=0.1, color='#575757')
        elif i == 11:
            plt.fill_between(x=range(len(mean)), y1=mean - std * STD_MULT, y2=mean + std * STD_MULT, alpha=0.2, color='#EBBB00')
            # plt.fill_between(x=range(len(mean)), y1=mean - std * STD_MULT, y2=mean + std * STD_MULT, alpha=0.1, color='#EBBB00')
        else:
            plt.fill_between(x=range(len(mean)), y1=mean - std * STD_MULT, y2=mean + std * STD_MULT, alpha=0.2)
            # plt.fill_between(x=range(len(mean)), y1=mean - std * STD_MULT, y2=mean + std * STD_MULT, alpha=0.1)

    leg = plt.legend(LEGEND, fontsize=7)

    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.title('HalfCheetah-v3 Transfer')
    plt.xlabel('steps')
    plt.xlim(0, MIN_STEPS)
    # plt.ylim(-1000, 6000)
    plt.ylim(-1000, 3000)
    plt.ylabel('cumulative reward')
    base_dir = os.path.dirname(LOG_FILES[0])
    plt.savefig(os.path.join(base_dir, savefig_name))
    plt.show()


if __name__ == "__main__":
    proc_data = get_data()
    if AUC:
        # plot_data(proc_data=proc_data, savefig_name=f'halfcheetah_auc_transfer_algo.pdf')
        plot_data(proc_data=proc_data, savefig_name=f'halfcheetah_auc_transfer_algo.png')
    else:
        # plot_data(proc_data=proc_data, savefig_name=f'halfcheetah_transfer_algo.pdf')
        plot_data(proc_data=proc_data, savefig_name=f'halfcheetah_transfer_algo.png')
