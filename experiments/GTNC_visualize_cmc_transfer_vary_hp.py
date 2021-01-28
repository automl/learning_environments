import matplotlib.pyplot as plt
import torch
import numpy as np

LOG_FILES = ['../results/cmc_compare_reward_envs/best_transfer_vary_hp1.pt',
             '../results/cmc_compare_reward_envs/best_transfer_vary_hp2.pt',
             '../results/cmc_compare_reward_envs/best_transfer_vary_hp5.pt',
             '../results/cmc_compare_reward_envs/best_transfer_vary_hp6.pt',
             '../results/cmc_compare_reward_envs/best_transfer_vary_hp0.pt']

STD_MULT = 0.2
MIN_STEPS = 100000

def get_data():
    list_data = []
    for log_file in LOG_FILES:
        data = torch.load(log_file)
        list_data.append((data['reward_list'], data['episode_length_list']))
        model_num = data['model_num']
        model_agents = data['model_agents']

    min_steps = float('Inf')
    # get minimum number of evaluations
    for reward_list, episode_length_list in list_data:
        for episode_lengths in episode_length_list:
            print(sum(episode_lengths))
            min_steps = min(min_steps, sum(episode_lengths))

    min_steps = max(min_steps, MIN_STEPS)
    # convert data from episodes to steps
    proc_data = []

    for reward_list, episode_length_list in list_data:
        np_data = np.zeros([model_num*model_agents,min_steps])

        for it, data in enumerate(zip(reward_list, episode_length_list)):
            rewards, episode_lengths = data

            concat_list = []
            rewards = rewards

            for i in range(len(episode_lengths)):
                concat_list += [rewards[i]]*episode_lengths[i]

            while len(concat_list) < min_steps:
                concat_list.append(concat_list[-1])

            np_data[it] = np.array(concat_list[:min_steps])

        mean = np.mean(np_data,axis=0)
        std = np.std(np_data,axis=0)

        proc_data.append((mean,std))

    return proc_data


def plot_data(proc_data, savefig_name):
    fig, ax = plt.subplots(dpi=600, figsize=(5,4))
    colors = []
    #
    # for mean, _ in data_w:
    #     plt.plot(mean_w)
    #     colors.append(plt.gca().lines[-1].get_color())

    for mean, std in proc_data:
        plt.plot(mean)

    for mean, std in proc_data:
        plt.fill_between(x=range(len(mean)), y1=mean - std * STD_MULT, y2=mean + std * STD_MULT, alpha=0.1)

    plt.legend(('TD3 + exc. pot. RN', 'TD3 + add. pot. RN', 'TD3 + exc. non-pot. RN', 'TD3 + add. non-pot. RN', 'TD3'), fontsize=7)
    #plt.xlim(0,99)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.title('MountainCarContinuous-v0 vary hyperparameters')
    plt.xlabel('steps')
    plt.xlim(0,100000)
    plt.ylabel('cumulative reward')
    plt.savefig(savefig_name)
    plt.show()

if __name__ == "__main__":
    proc_data = get_data()
    plot_data(proc_data=proc_data, savefig_name='cmc_transfer_vary_hp.png')



