import numpy as np
import torch
from scipy import stats

file_path_ddqn = "/home/ferreira/Projects/learning_environments/experiments/correlation_hps_se_real/2_ddqn_vary_correlation_syn_real_10_agents_num_40_model_num.pt"
file_path_duelingddqn = "/home/ferreira/Projects/learning_environments/experiments/correlation_hps_se_real/2_duelingddqn_vary_correlation_syn_real_10_agents_num_40_model_num.pt"

rewards_ddqn = torch.load(file_path_ddqn)
rewards_duelingddqn = torch.load(file_path_duelingddqn)

rewards_ddqn_real = []
rewards_ddqn_synth = []

rewards_duel_ddqn_real = []
rewards_duel_ddqn_synth = []

for i, r in enumerate(rewards_ddqn['reward_list']):
    # if i > 5:
    #     break
    for k, v in r.items():
        if k == "real":
            flat_list_real = [item for sublist in v for item in sublist]
            # rewards_ddqn_real.append(flat_list_real)
            rewards_ddqn_real.append(np.mean(flat_list_real))
        elif k == "synthetic":
            flat_list_syn = [item for sublist in v for item in sublist]
            # rewards_ddqn_synth.append(flat_list_syn)
            rewards_ddqn_synth.append(np.mean(flat_list_syn))

for i, r in enumerate(rewards_duelingddqn['reward_list']):
    # if i > 5:
    #     break
    for k, v in r.items():
        if k == "real":
            flat_list_real = [item for sublist in v for item in sublist]
            # rewards_duel_ddqn_real.append(flat_list_real)
            rewards_duel_ddqn_real.append(np.mean(flat_list_real))
        elif k == "synthetic":
            flat_list_syn = [item for sublist in v for item in sublist]
            # rewards_duel_ddqn_synth.append(flat_list_syn)
            rewards_duel_ddqn_synth.append(np.mean(flat_list_syn))

# model-wise
# rho, pval = stats.spearmanr(np.asarray(rewards_ddqn_real), np.asarray(rewards_ddqn_synth), axis=1) # if axis=1, the relationship is transposed: each row represents a variable, while the columns contain observations
# print(f"model-wise spearman rank corr.; rho: {rho}, pval: {pval}")

# sns.heatmap(rho, annot=True)
# plt.show()

# DDQN
rho, pval = stats.spearmanr(np.asarray(rewards_ddqn_real).flatten(), np.asarray(rewards_ddqn_synth).flatten())
print(f"DDQN Spearman rank corr.; rho: {rho}, pval: {pval}")

rho, pval = stats.kendalltau(np.asarray(rewards_ddqn_real).flatten(), np.asarray(rewards_ddqn_synth).flatten())
print(f"DDQN Kendalltau corr.; rho: {rho}, pval: {pval}")

rho, pval = stats.pearsonr(np.asarray(rewards_ddqn_real).flatten(), np.asarray(rewards_ddqn_synth).flatten())
print(f"DDQN Pearson corr.; rho: {rho}, pval: {pval}")

# Dueling DDQN
rho, pval = stats.spearmanr(np.asarray(rewards_duel_ddqn_real).flatten(), np.asarray(rewards_duel_ddqn_synth).flatten())
print(f"DuelingDDQN Spearman rank corr.; rho: {rho}, pval: {pval}")

rho, pval = stats.kendalltau(np.asarray(rewards_duel_ddqn_real).flatten(), np.asarray(rewards_duel_ddqn_synth).flatten())
print(f"DuelingDDQN Kendalltau corr.; rho: {rho}, pval: {pval}")

rho, pval = stats.pearsonr(np.asarray(rewards_duel_ddqn_real).flatten(), np.asarray(rewards_duel_ddqn_synth).flatten())
print(f"DuelingDDQN Pearson corr.; rho: {rho}, pval: {pval}")
