import matplotlib.pyplot as plt
import numpy as np
import torch

LOG_FILES = ['../results/cartpole_compare_reward_envs/best_transfer_vary_hp1.pt',
             '../results/cartpole_compare_reward_envs/best_transfer_vary_hp2.pt',
             '../results/cartpole_compare_reward_envs/best_transfer_vary_hp5.pt',
             '../results/cartpole_compare_reward_envs/best_transfer_vary_hp6.pt',
             '../results/cartpole_compare_reward_envs/best_transfer_vary_hp0.pt',
             '../results/cartpole_compare_reward_envs/best_transfer_vary_hp-1.pt']

STD_MULT = 0.2
MIN_STEPS = 10000


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

            np_data[it] = np.array(concat_list[:min_steps])

        mean = np.mean(np_data, axis=0)
        std = np.std(np_data, axis=0)

        proc_data.append((mean, std))

    return proc_data


def plot_data(proc_data, savefig_name):
    fig, ax = plt.subplots(dpi=600, figsize=(5, 4))

    for mean, std in proc_data:
        plt.plot(mean)

    for mean, std in proc_data:
        plt.fill_between(x=range(len(mean)), y1=mean - std * STD_MULT, y2=mean + std * STD_MULT, alpha=0.1)

    plt.legend(('DDQN + exc. pot. RN', 'DDQN + add. pot. RN', 'DDQN + exc. non-pot. RN', 'DDQN + add. non-pot. RN', 'DDQN', 'DDQN + ICM'),
               fontsize=7)

    # plt.xlim(0,99)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.title('CartPole-v0 varied hyperparameters')
    plt.xlabel('steps')
    plt.xlim(0, MIN_STEPS)
    plt.ylim(0, 210)
    plt.ylabel('cumulative reward')
    plt.savefig(savefig_name)
    plt.show()


if __name__ == "__main__":
    proc_data = get_data()
    plot_data(proc_data=proc_data, savefig_name='cartpole_transfer_vary_hp.png')
