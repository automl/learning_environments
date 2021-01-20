import matplotlib.pyplot as plt
import torch
import numpy as np

LOG_FILES = ['../results/gridworld_compare_reward_envs/best20_0.pt',
             '../results/gridworld_compare_reward_envs/best20_2.pt',
             '../results/gridworld_compare_reward_envs/best20_5.pt']

STD_MULT = 0.5
MAX_VALS = 60

def get_data():
    list_data = []
    for log_file in LOG_FILES:
        data = torch.load(log_file)
        list_data.append(data['reward_list'])

    # copy from list to numpy array
    proc_data = []

    n = len(list_data[0][0])
    for data in list_data:
        np_data = np.zeros([MAX_VALS,n])

        for i in range(len(np_data)):
            np_data[i] = np.array(data[i])

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

    plt.legend(('baseline naive','mode 2', 'mode 5'))
    plt.xlim(0,99)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.title('HoleRoomLarge')
    plt.xlabel('episode')
    plt.ylabel('average reward')
    plt.savefig(savefig_name)
    plt.show()

if __name__ == "__main__":
    proc_data = get_data()
    plot_data(proc_data=proc_data, savefig_name='gridworld_compare_reward_env.png')



