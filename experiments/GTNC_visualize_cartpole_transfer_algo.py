import matplotlib.pyplot as plt
import torch
import numpy as np

LOG_FILES = ['../results/cartpole_compare_reward_envs/best_transfer_algo1.pt',
             '../results/cartpole_compare_reward_envs/best_transfer_algo2.pt',
             '../results/cartpole_compare_reward_envs/best_transfer_algo5.pt',
             '../results/cartpole_compare_reward_envs/best_transfer_algo6.pt',
             '../results/cartpole_compare_reward_envs/best_transfer_algo0.pt']

STD_MULT = 0.2

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
            min_steps = min(min_steps, sum(episode_lengths))


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

            np_data[it] = np.array(concat_list[:min_steps])

        mean = np.mean(np_data,axis=0)
        std = np.std(np_data,axis=0)

        proc_data.append((mean,std))

    return proc_data


def plot_data(proc_data, savefig_name):
    fig, ax = plt.subplots(dpi=600, figsize=(5,4))

    for mean, std in proc_data:
        plt.plot(mean)

    for mean, std in proc_data:
        plt.fill_between(x=range(len(mean)), y1=mean - std * STD_MULT, y2=mean + std * STD_MULT, alpha=0.1)

    plt.legend(('Duel.-DDQN + exc. pot. RN', 'Duel.-DDQN + add. pot. RN', 'Duel.-DDQN + exc. non-pot. RN', 'Duel.-DDQN + add. non-pot. RN', 'Duel.-DDQN'), fontsize=7)
    #plt.xlim(0,99)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.title('CartPole-v0 transferred algorithm')
    plt.xlabel('steps')
    plt.xlim(0,10000)
    plt.ylim(0,210)
    plt.ylabel('cumulative reward')
    plt.savefig(savefig_name)
    plt.show()

if __name__ == "__main__":
    proc_data = get_data()
    plot_data(proc_data=proc_data, savefig_name='cartpole_transfer_algo.png')




