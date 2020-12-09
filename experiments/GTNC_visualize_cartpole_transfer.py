import os
import torch
import statistics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#FILE_DIR = '/home/dingsda/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_vary_hp_2020-11-17-10/GTN_models_CartPole-v0'
#FILE_DIR = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_vary_hp_2020-11-17-10/GTN_models_CartPole-v0'
FILE_DIR = '/home/ferreira/Projects/learning_environments/results/DDQN_to_DuelingDDQN_CartPole_transfer'
FILE_LIST = ['0.pt', '2.pt', '1.pt']


if __name__ == "__main__":
    data_list = []
    mean_list = []
    for file in FILE_LIST:
        file_path = os.path.join(FILE_DIR, file)
        save_dict = torch.load(file_path)
        reward_list = save_dict['reward_list']

        reward_list_single = []
        for r_list in reward_list:
            reward_list_single += r_list

        data_list.append(reward_list_single)
        mean_list.append('mean: ' + str(statistics.mean(reward_list_single)))

    data_dict = {'train: real  / HP: vary': data_list[0],
                 'train: synth. / HP: vary': data_list[1],
                 'train: synth. / HP: no vary': data_list[2]}

    df = pd.DataFrame(data=data_dict)
    plt.figure(dpi=600, figsize=(7.5,3))
    ax = sns.violinplot(data=df, cut=0, inner=None)
    plt.ylabel('cumulative reward')

    for x,y,mean in zip([0,1,2], [220,220,220], mean_list):
        plt.text(x,y,mean, ha='center', va='center')

    plt.savefig('ddqn_to_dueling_ddqn_cartpole_vary_hp_transfer.svg', bbox_inches='tight')
    plt.show()


