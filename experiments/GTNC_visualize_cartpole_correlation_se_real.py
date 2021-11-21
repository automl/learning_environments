import numpy as np
import torch
import os

file_path_ddqn = "/home/ferreira/Projects/learning_environments/experiments/correlation_hps_se_real/2_ddqn_vary_correlation_syn_real_10_agents_num_40_model_num.pt"
file_path_duelingddqn = "/home/ferreira/Projects/learning_environments/experiments/correlation_hps_se_real/2_duelingddqn_vary_correlation_syn_real_10_agents_num_40_model_num.pt"

rewards_ddqn = torch.load(file_path_ddqn)
print(rewards_ddqn)