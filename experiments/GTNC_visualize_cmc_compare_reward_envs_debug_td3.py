import matplotlib.pyplot as plt
import torch
import numpy as np
from datetime import datetime
import glob

LOG_FILES = []
# os.chdir("../results/debug_cmc_td3")
for file in glob.glob("../results/debug_cmc_td3/*.pt"):
    print(file)
    LOG_FILES.append(file)

# LOG_FILES = [
#              # '../results/debug_cmc_td3/best-1.pt',
#              '../results/debug_cmc_td3/best0.pt'
#              ]

STD_MULT = 0.2
# STD_MULT = 1.
MIN_STEPS = 100000

IMPORTANT_KEYS = ["action_std", "activation_fn", "batch_size", "gamma", "lr", "policy_delay", "policy_std", "policy_std_clip", "tau",
                  "same_action_num", "rb_size"]

# IMPORTANT_KEYS = ["activation_fn", "lr", "policy_delay", "policy_std_clip", "tau",
#                   "same_action_num", "rb_size"]

def get_data():
    list_data = []
    list_hyperparameters = []
    for log_file in LOG_FILES:
        file_name_hyperparameters = ""
        data = torch.load(log_file)
        list_data.append((data['reward_list'], data['episode_length_list']))
        model_num = data['model_num']
        model_agents = data['model_agents']
        for k, v in data["config"]["agents"]["td3"].items():
            if k in IMPORTANT_KEYS:
                file_name_hyperparameters += f"{k}: {v} "

        list_hyperparameters.append(file_name_hyperparameters)


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

    return proc_data, list_hyperparameters


def plot_data(proc_data, savefig_name, list_hyperparameters):
    # fig, ax = plt.subplots(dpi=600, figsize=(10,10))
    # colors = []
    #
    # for mean, _ in data_w:
    #     plt.plot(mean_w)
    #     colors.append(plt.gca().lines[-1].get_color())
    f = plt.figure(figsize=(15, 15))
    ax = f.add_subplot(111)

    for mean, std in proc_data:
        plt.plot(mean)

    for mean, std in proc_data:
        plt.fill_between(x=range(len(mean)), y1=mean - std * STD_MULT, y2=mean + std * STD_MULT, alpha=0.1)
    # plt.legend(loc=(1.04, 0))
    plt.legend(list_hyperparameters, fontsize=9, loc='best')
    plt.tight_layout()
    # plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    # plt.subplots_adjust(right=0.7)
    # plt.legend(('TD3'), fontsize=7)
    #plt.xlim(0,99)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.title('MountainCarContinuous-v0')
    plt.xlabel('steps')
    plt.xlim(0,100000)
    # plt.xlim(0, 60000)
    plt.ylim(-75,100)
    plt.ylabel('cumulative reward')
    plt.savefig(savefig_name)
    plt.show()

if __name__ == "__main__":
    proc_data, list_hyperparameters = get_data()
    time = datetime.now().strftime("%Y_%m_%d_%I_%M_%S")
    file_name = "../results/debug_cmc_td3/" + time + "_cmc_compare_reward_env.png"
    plot_data(proc_data=proc_data, savefig_name=file_name, list_hyperparameters=list_hyperparameters)



