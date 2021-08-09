import argparse
import multiprocessing as mp
import os

import numpy as np
import torch
import yaml

from agents.agent_utils import select_agent
from envs.env_factory import EnvFactory
from experiments.syn_env_run_vary_hp import run_vary_hp
from utils import save_lists
from torch.utils.data import TensorDataset, DataLoader

MBRL_PATH = "../mbrl_baseline"


def get_saved_data(mypath):
    f_names = []
    gzip_files = []
    for (dirpath, dirnames, filename) in os.walk(mypath):
        f_names.extend(filename)
        break
    for name in f_names:
        if name[-4:] == "gzip":
            gzip_files.append(name)
    return gzip_files


def get_data_for_each_bbhob_run(data_filename_list):
    # TODO: sort names and make a list for each bohb run
    pass

def data_to_xy_data(data:list):
    pass

def get_data_loader(train_x, train_y, batchsize=1024, shuffle=False):
    # TODO: process data into x and y
    data_set = TensorDataset(train_x, train_y)
    train_dataloader = DataLoader(data_set, batch_size=batchsize, shuffle=shuffle)
    return train_dataloader

def load_env_wrapped():
    with open("default_config_cartpole.yaml", 'r') as stream:
        default_config = yaml.safe_load(stream)
    env_factory = EnvFactory(config=default_config)
    mbrl_baseline = env_factory.generate_virtual_env()
    return mbrl_baseline


def get_env_internal_models(env_wrapped):
    state_net = env_wrapped.state_net
    reward_net = env_wrapped.reward_net
    done_net = env_wrapped.done_net
    return state_net, reward_net, done_net
    

def fit_mbrl_baseline():
        pass

def evaluate_mbrl_baseline(mbrl_baseline, real_env, config, pool, agents_num, experiment_name):
    env_reward_overview = {}
    reward_list = []
    train_steps_needed = []
    episode_length_needed = []

    reward_list_tpl, train_steps_needed_tpl, episode_length_needed_tpl = zip(*pool.starmap(train_test_agent,
                                                                                           [(mbrl_baseline, real_env, config)
                                                                                            for _ in range(agents_num)]
                                                                                           )
                                                                             )
    # starmap/map preservers order of calling
    for i in range(agents_num):
        reward_list += reward_list_tpl[i]
        train_steps_needed += train_steps_needed_tpl[i]
        episode_length_needed += episode_length_needed_tpl[i]
        env_reward_overview[real_env.env.env_name + "_" + str(i)] = np.hstack(reward_list_tpl[i])
    
    save_lists(mode=3,  # mbrl baseline
               config=config,
               reward_list=reward_list,
               train_steps_needed=train_steps_needed,
               episode_length_needed=episode_length_needed,
               env_reward_overview=env_reward_overview,
               experiment_name=experiment_name
               )

# def load_envs_and_config(file_name, model_dir, device):
#     file_path = os.path.join(model_dir, file_name)
#     save_dict = torch.load(file_path)
#     config = save_dict['config']
#     config['device'] = device
#     env_factory = EnvFactory(config=config)
#     virtual_env = env_factory.generate_virtual_env()
#     virtual_env.load_state_dict(save_dict['model'])
#     real_env = env_factory.generate_real_env()
#
#     return virtual_env, real_env, config


def train_test_agent(train_env, test_env, config, agent_name):
    reward_list = []
    train_steps_needed = []
    episodes_needed = []
    
    # settings for comparability
    if agent_name == "ddqn_vary":
        config['agents']['ddqn_vary']['vary_hp'] = True
        config['agents']['ddqn']['print_rate'] = 10
        config['agents']['ddqn']['early_out_num'] = 10
        config['agents']['ddqn']['train_episodes'] = 1000
        config['agents']['ddqn']['init_episodes'] = 10
        config['agents']['ddqn']['test_episodes'] = 10
        config['agents']['ddqn']['early_out_virtual_diff'] = 0.01

    elif agent_name == "duelingddqn_vary":
        config['agents']['duelingddqn_vary']['vary_hp'] = True
        config['agents']['duelingddqn']['print_rate'] = 10
        config['agents']['duelingddqn']['early_out_num'] = 10
        config['agents']['duelingddqn']['train_episodes'] = 1000
        config['agents']['duelingddqn']['init_episodes'] = 10
        config['agents']['duelingddqn']['test_episodes'] = 10
        config['agents']['duelingddqn']['early_out_virtual_diff'] = 0.01
        
    elif agent_name == "td3_discrete_vary":
        config['agents']['td3_discrete_vary']['vary_hp'] = True
        config['agents']['td3_discrete_vary']['print_rate'] = 10
        config['agents']['td3_discrete_vary']['early_out_num'] = 10
        config['agents']['td3_discrete_vary']['train_episodes'] = 1000
        config['agents']['td3_discrete_vary']['init_episodes'] = 10
        config['agents']['td3_discrete_vary']['test_episodes'] = 10
        config['agents']['td3_discrete_vary']['early_out_virtual_diff'] = 0.01
    else:
        raise ValueError("wrong agent_name")

    agent = select_agent(config=config, agent_name=agent_name)
    reward_train, episode_length, _ = agent.train(env=train_env)
    reward, _, _ = agent.test(env=test_env)
    print('reward: ' + str(reward))
    reward_list.append(reward)
    train_steps_needed.append([sum(episode_length)])
    episodes_needed.append([(len(reward_train))])
    
    return reward_list, train_steps_needed, episodes_needed


if __name__ == "__main__":
    debug = True
    
    if debug:
        data_names_list = get_saved_data(mypath=MBRL_PATH)
        print(data_names_list)
    else:
        parser = argparse.ArgumentParser()
        # parser.add_argument('--mode', type=int, help='mode 0: real env, mode 1: syn. env. (no vary), mode 2: syn. env. (vary)')
        parser.add_argument("--agent_name", type=str, help="must be in [ddqn_vary, duelingddqn_vary, td3_discrete_vary]", default="ddqn_vary")
        parser.add_argument('--pool', type=int, help='size of the multiprocessing pool')
        parser.add_argument('--agents_num', type=int, help='number of agents evaluated', default=10)
        parser.add_argument('--device', type=str, help='device to be used (cuda or cpu)', default='cpu')
        args = parser.parse_args()
        
        model_num = args.model_num
        agents_num = args.agents_num
        device = args.device
        agent_name = args.agent_name
        
        print("model_num:", model_num, "agents_num:", agents_num, "pool size:", args.pool, "device:", device)
        
        experiment_name = f"mbrl_baseline_{agent_name}_transfer_agents_num_{agents_num}"
        
        env_name = "CartPole"
        
        if args.pool is not None:
            mp.set_start_method('spawn')  # due to cuda
            pool = mp.Pool(args.pool)
        else:
            raise ValueError("pool not instantiated")
    
        mbrl_baseline, real_env, config = fit_mbrl_baseline()
        evaluate_mbrl_baseline(mbrl_baseline, real_env, config, pool, agents_num, experiment_name)