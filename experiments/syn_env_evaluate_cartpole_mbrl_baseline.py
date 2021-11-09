import argparse
import multiprocessing as mp
import os

import torch
import yaml

from agents.agent_utils import select_agent
from envs.env_factory import EnvFactory
from experiments.syn_env_run_vary_hp import run_vary_hp

MBRL_PATH = "../mbrl_baseline"


def fit_mbrl_baseline():
    with open("default_config_cartpole.yaml", 'r') as stream:
        default_config = yaml.safe_load(stream)
    env_factory = EnvFactory(config=default_config)
    synthetic_env = env_factory.generate_virtual_env()

    raise NotImplementedError


def evaluate_mbrl_baseline(mbrl_model):
    with open("default_config_cartpole.yaml", 'r') as stream:
        default_config = yaml.safe_load(stream)

    env_factory = EnvFactory(config=default_config)
    synthetic_env = env_factory.generate_virtual_env()
    synthetic_env.load_state_dict(mbrl_model.parameters())
    real_env = env_factory.generate_real_env()


def load_envs_and_config(file_name, model_dir, device):
    file_path = os.path.join(model_dir, file_name)
    save_dict = torch.load(file_path)
    config = save_dict['config']
    config['device'] = device
    env_factory = EnvFactory(config=config)
    virtual_env = env_factory.generate_virtual_env()
    virtual_env.load_state_dict(save_dict['model'])
    real_env = env_factory.generate_real_env()

    return virtual_env, real_env, config


def train_test_agents(train_env, test_env, config, agents_num):
    reward_list = []
    train_steps_needed = []
    episodes_needed = []

    # settings for comparability
    config['agents']['ddqn_vary']['vary_hp'] = True
    config['agents']['ddqn']['print_rate'] = 10
    config['agents']['ddqn']['early_out_num'] = 10
    config['agents']['ddqn']['train_episodes'] = 1000
    config['agents']['ddqn']['init_episodes'] = 10
    config['agents']['ddqn']['test_episodes'] = 10
    config['agents']['ddqn']['early_out_virtual_diff'] = 0.01

    for i in range(agents_num):
        agent = select_agent(config=config, agent_name='DDQN_vary')
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

    experiment_name = "ddqn_vary_transfer_reward_overview_" + \
                      str(agents_num) + "_agents_num_" + str(model_num) + "_model_num"

    env_name = "CartPole"

    if args.pool is not None:
        mp.set_start_method('spawn')  # due to cuda
        pool = mp.Pool(args.pool)
    else:
        pool = None

    if args.mode is not None:
        run_vary_hp(
            mode=args.mode, experiment_name=experiment_name, model_num=model_num, agents_num=agents_num,
            model_dir=model_dir, custom_load_envs_and_config=load_envs_and_config,
            custom_train_test_agents=train_test_agents, env_name=env_name, pool=pool, device=device
        )
    else:
        for mode in range(3):
            run_vary_hp(
                mode=mode, experiment_name=experiment_name, model_num=model_num, agents_num=agents_num,
                model_dir=model_dir, custom_load_envs_and_config=load_envs_and_config,
                custom_train_test_agents=train_test_agents, env_name=env_name, pool=pool, device=device
            )
