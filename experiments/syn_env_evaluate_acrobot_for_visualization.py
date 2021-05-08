import argparse
import multiprocessing as mp
import os

import torch

from agents.agent_utils import select_agent
from envs.env_factory import EnvFactory
from experiments.syn_env_run_vary_hp import run_vary_hp


def load_envs_and_config(file_path, device):
    save_dict = torch.load(file_path)
    config = save_dict['config']
    config['device'] = device
    env_factory = EnvFactory(config=config)
    virtual_env = env_factory.generate_virtual_env()
    virtual_env.load_state_dict(save_dict['model'])
    real_env = env_factory.generate_real_env()

    return virtual_env, real_env, config


def train_test_agents(train_env, test_env, config):
    reward_list = []
    train_steps_needed = []
    episodes_needed = []

    config["agents"]["ddqn"]["print_rate"] = 1
    config["agents"]["ddqn"]["test_episodes"] = 10
    config["render_env"] = True

    agent = select_agent(config=config, agent_name='DDQN')
    reward_train, episode_length, _ = agent.train(env=train_env)
    reward, _, _ = agent.test(env=test_env)
    print('reward: ' + str(reward))
    reward_list.append(reward)
    train_steps_needed.append([sum(episode_length)])
    episodes_needed.append([(len(reward_train))])

    return reward_list, train_steps_needed, episodes_needed


if __name__ == "__main__":
    file_path = '/home/dingsda/master_thesis/learning_environments/results/GTNC_evaluate_acrobot_2020-11-28-16/GTN_models_Acrobot-v1_opt_200/Acrobot-v1_13_BSLPEB.pt'

    virtual_env, real_env, config = load_envs_and_config(file_path, device='cuda')
    train_test_agents(train_env=virtual_env, test_env=real_env, config=config)