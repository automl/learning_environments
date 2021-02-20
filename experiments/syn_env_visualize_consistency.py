import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd

# FILE_DIR = "consistency_experiments/cartpole"
# FILE_LIST = [
#              '2_ddqn_vary_transfer_reward_overview_1000_agents_num_40_model_num.pt',
#              '2_ddqn_to_duelingddqn_vary_transfer_reward_overview_1000_agents_num_40_model_num.pt',
#              '2_ddqn_to_td3_discrete_vary_transfer_reward_overview_1000_agents_num_40_model_num.pt',
#              ]

FILE_DIR = "consistency_experiments/acrobot"
FILE_LIST = [
             '2_ddqn_vary_acrobot_reward_overview_1000_agents_num_40_model_num.pt',
             ]

if __name__ == "__main__":
    env_reward_overviews = []
    for file in FILE_LIST:
        file_path = os.path.join(FILE_DIR, file)
        save_dict = torch.load(file_path)
        env_reward_overview = save_dict['env_reward_overview']

        new_dct = {}
        for k, v in env_reward_overview.transpose().to_dict().items():
            rewards = []
            for values in v.values():
                rewards.append(values)
            new_dct[k] = {
                    "name": k,
                    "rewards": rewards
                    }

        df = pd.DataFrame.from_dict(data=new_dct, columns=["name", "rewards"], orient="index").explode("rewards").reset_index(drop=True)
        df["rewards"] = pd.to_numeric(df["rewards"])

        # fig, ax = plt.subplots()
        fig, ax = plt.subplots(figsize=(30, 10))
        axis = sns.violinplot(x="name", y="rewards", data=df, axes=ax, cut=0, inner=None, scale='width')
        # axis = sns.violinplot(x="name", y="rewards", data=df, axes=ax, cut=0, inner=None)
        axis.set_xticklabels(rotation=90, labels=axis.get_xticklabels())
        axis.set_xlabel('model')
        axis.set_ylabel('cumulative reward')
        plot_name = Path(file_path).stem
        plt.savefig(os.path.join(FILE_DIR, plot_name + "scaled_width.eps"), bbox_inches='tight')
        # plt.savefig(os.path.join(FILE_DIR, plot_name + ".eps"), bbox_inches='tight')
        plt.show()
