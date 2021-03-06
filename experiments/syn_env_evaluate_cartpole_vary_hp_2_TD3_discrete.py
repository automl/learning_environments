import argparse
import multiprocessing as mp
import os

import torch
import yaml

from agents.agent_utils import select_agent
from envs.env_factory import EnvFactory
from experiments.syn_env_run_vary_hp import run_vary_hp

filter_models_list = ["CartPole-v0_25_4I271D.pt", "CartPole-v0_22_CW9CSH.pt", "CartPole-v0_21_DRWNIK.pt", "CartPole-v0_8_QGN9IF.pt",
                      "CartPole-v0_39_VICT5Q.pt", "CartPole-v0_17_WPWK3R.pt"]

# 5 models
# filter_models_list = ["CartPole-v0_25_4I271D.pt", "CartPole-v0_22_CW9CSH.pt", "CartPole-v0_21_DRWNIK.pt", "CartPole-v0_8_QGN9IF.pt",
#                       "CartPole-v0_17_WPWK3R.pt"]

mediocre = ["CartPole-v0_9_CM4I44.pt", "CartPole-v0_37_5N1W77.pt", "CartPole-v0_18_F54MTK.pt", "CartPole-v0_9_U821ST.pt"]

filter_models_list += mediocre

def load_envs_and_config(file_name, model_dir, device):
    file_path = os.path.join(model_dir, file_name)
    save_dict = torch.load(file_path)
    config = save_dict['config']
    config['device'] = device
    env_factory = EnvFactory(config=config)
    virtual_env = env_factory.generate_virtual_env()
    virtual_env.load_state_dict(save_dict['model'])
    real_env = env_factory.generate_real_env()

    # load additional agent configs
    with open("../default_config_cartpole.yaml", "r") as stream:
        config_new = yaml.safe_load(stream)["agents"]

    # config["agents"]["td3_discrete_vary"] = config_new["td3_discrete_temp_anneal"]
    config["agents"]["td3_discrete_vary"] = config_new["td3_discrete_vary_layer_norm_2"]

    return virtual_env, real_env, config


def train_test_agents(train_env, test_env, config, agents_num):
    reward_list = []
    train_steps_needed = []
    episodes_needed = []

    # settings for comparability
    config['agents']['td3_discrete_vary']['vary_hp'] = True
    config['agents']['td3_discrete_vary']['print_rate'] = 10
    config['agents']['td3_discrete_vary']['early_out_num'] = 10
    config['agents']['td3_discrete_vary']['train_episodes'] = 1000
    config['agents']['td3_discrete_vary']['init_episodes'] = 10
    config['agents']['td3_discrete_vary']['test_episodes'] = 10
    config['agents']['td3_discrete_vary']['early_out_virtual_diff'] = 0.01

    for i in range(agents_num):
        agent = select_agent(config=config, agent_name='td3_discrete_vary')
        reward_train, episode_length, _ = agent.train(env=train_env)
        reward, _, _ = agent.test(env=test_env)
        print('reward: ' + str(reward))
        reward_list.append(reward)
        train_steps_needed.append([sum(episode_length)])
        episodes_needed.append([(len(reward_train))])

    return reward_list, train_steps_needed, episodes_needed


if __name__ == "__main__":
    model_dir = '/home/nierhoff/master_thesis/learning_environments/results/GTNC_evaluate_cartpole_vary_hp_2020-11-17-10' \
                '/GTN_models_CartPole-v0'

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, help='mode 0: real env, mode 1: syn. env. (no vary), mode 2: syn. env. (vary)')
    parser.add_argument('--pool', type=int, help='size of the multiprocessing pool')
    parser.add_argument('--agents_num', type=int, help='number of agents evaluated', default=10)
    parser.add_argument('--model_num', type=int, help='number of models evaluated', default=40)
    parser.add_argument('--device', type=str, help='device to be used (cuda or cpu)', default='cuda')
    args = parser.parse_args()

    model_num = args.model_num
    agents_num = args.agents_num
    device = args.device

    print("model_num:", model_num, "agents_num:", agents_num, "pool size:", args.pool, "device:", device)

    experiment_name = "ddqn_to_td3_discrete_vary_transfer_reward_overview_only_10_best_filtered_models_4000_evals_" + \
                      str(agents_num) + "_agents_num_" + str(model_num) + "_model_num"

    env_name = "CartPole"

    if args.pool is not None:
        mp.set_start_method('spawn')  # due to cuda
        pool = mp.Pool(args.pool)
    else:
        pool = None

    if args.mode is not None:
        run_vary_hp(mode=args.mode, experiment_name=experiment_name, model_num=model_num, agents_num=agents_num,
                    model_dir=model_dir, custom_load_envs_and_config=load_envs_and_config,
                    custom_train_test_agents=train_test_agents, env_name=env_name, pool=pool, device=device, filter_models_list=filter_models_list)
    else:
        for mode in range(3):
            run_vary_hp(mode=mode, experiment_name=experiment_name, model_num=model_num, agents_num=agents_num,
                        model_dir=model_dir, custom_load_envs_and_config=load_envs_and_config,
                        custom_train_test_agents=train_test_agents, env_name=env_name, pool=pool, device=device, filter_models_list=filter_models_list)
